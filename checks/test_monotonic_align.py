import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(_PROJECT_ROOT / "StyleTTS2"))

from utils import maximum_path


def make_block_mask(B, T_T, T_S, t_t_lens, t_s_lens):
    mask = torch.zeros(B, T_T, T_S)
    for b in range(B):
        mask[b, : t_t_lens[b], : t_s_lens[b]] = 1
    return mask


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------


def test_output_shape():
    B, T_T, T_S = 2, 4, 8
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    assert path.shape == (B, T_T, T_S)


def test_output_is_binary():
    B, T_T, T_S = 2, 4, 8
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    unique_vals = set(path.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0}), f"Non-binary values in path: {unique_vals}"


# ---------------------------------------------------------------------------
# Structural correctness
# ---------------------------------------------------------------------------


def test_every_frame_assigned_exactly_once():
    """Each mel frame within the valid region must be assigned to exactly one text token."""
    B, T_T, T_S = 2, 3, 6
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    col_sums = path.sum(dim=1)  # [B, T_S]
    assert (col_sums == 1).all(), f"Frame column sums (expected all 1):\n{col_sums}"


def test_every_token_gets_at_least_one_frame():
    """Every text token must be assigned at least one mel frame."""
    B, T_T, T_S = 2, 3, 9
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    row_sums = path.sum(dim=2)  # [B, T_T]
    assert (row_sums >= 1).all(), f"Token row sums:\n{row_sums}"


def test_alignment_is_monotonic():
    """
    The frame assigned to each token must be non-decreasing across tokens
    (no crossing alignments).
    """
    B, T_T, T_S = 3, 5, 15
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    # argmax over mel-frame dim gives the assigned frame index for each token
    assigned_frames = path.argmax(dim=2)  # [B, T_T]
    for b in range(B):
        frames = assigned_frames[b].tolist()
        assert frames == sorted(frames), f"Batch {b}: non-monotonic frame indices {frames}"


def test_assigned_token_ranges_are_contiguous():
    """Frames assigned to a given text token must form a contiguous block."""
    B, T_T, T_S = 2, 4, 8
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    for b in range(B):
        for t in range(T_T):
            ones = path[b, t].nonzero(as_tuple=True)[0]
            if len(ones) > 1:
                # Indices should be consecutive integers
                indices = ones.tolist()
                assert indices == list(range(indices[0], indices[-1] + 1)), (
                    f"Batch {b}, token {t}: non-contiguous frames {indices}"
                )


# ---------------------------------------------------------------------------
# Cost optimality
# ---------------------------------------------------------------------------


def test_path_maximises_neg_cent():
    """
    The algorithm maximises the total neg_cent along the path, consistent
    with the MAS formulation from the VITS paper (2005.11129) where
    neg_cent = log p(z | text) and the most likely alignment is selected.

    All cells are set to low (-100) except the cells forming the expected
    coverage path (+100).  This makes the expected path the unique maximum,
    since every alternative coverage requires at least one low-value cell.

    Expected alignment: token 0 → frames [0,1], token 1 → [2,3], token 2 → [4,5].
    """
    B, T_T, T_S = 1, 3, 6
    neg_cent = torch.full((B, T_T, T_S), -100.0)
    high = [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)]
    for t, s in high:
        neg_cent[0, t, s] = 100.0

    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)

    expected = torch.zeros(B, T_T, T_S)
    for t, s in high:
        expected[0, t, s] = 1.0

    assert (path == expected).all(), f"Unexpected path:\n{path[0]}\nExpected:\n{expected[0]}"


def test_single_token_covers_all_frames():
    """Edge case: T_T=1 → single token must claim every mel frame."""
    B, T_T, T_S = 1, 1, 5
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)
    assert path[0, 0].sum().item() == T_S, "Single token must cover all frames"


def test_equal_tokens_and_frames():
    """Edge case: T_T == T_S → one-to-one mapping along the diagonal."""
    B, T_T, T_S = 1, 4, 4
    # Make the strict diagonal the cheapest option
    neg_cent = torch.full((B, T_T, T_S), 50.0)
    for i in range(T_T):
        neg_cent[0, i, i] = -50.0

    mask = torch.ones(B, T_T, T_S)
    path = maximum_path(neg_cent, mask)

    expected = torch.eye(T_T).unsqueeze(0)
    assert (path == expected).all(), f"Expected identity alignment:\n{path[0]}"


# ---------------------------------------------------------------------------
# Mask handling
# ---------------------------------------------------------------------------


def test_path_zero_outside_mask():
    """Path entries outside the valid (t_t, t_s) block must be zero."""
    B, T_T, T_S = 2, 5, 10
    t_t_lens = [3, 5]
    t_s_lens = [6, 10]
    neg_cent = torch.randn(B, T_T, T_S)
    mask = make_block_mask(B, T_T, T_S, t_t_lens, t_s_lens)
    path = maximum_path(neg_cent, mask)

    for b in range(B):
        assert (path[b, t_t_lens[b] :, :] == 0).all(), (
            f"Batch {b}: path non-zero beyond t_t boundary"
        )
        assert (path[b, :, t_s_lens[b] :] == 0).all(), (
            f"Batch {b}: path non-zero beyond t_s boundary"
        )


def test_batch_items_are_independent():
    """
    Changing neg_cent for one batch item must not affect another batch item's
    path.
    """
    B, T_T, T_S = 2, 3, 6
    neg_cent = torch.randn(B, T_T, T_S)
    mask = torch.ones(B, T_T, T_S)

    path_joint = maximum_path(neg_cent, mask)

    # Run each batch item in isolation
    for b in range(B):
        single_neg = neg_cent[b : b + 1]
        single_mask = mask[b : b + 1]
        path_single = maximum_path(single_neg, single_mask)
        assert (path_joint[b : b + 1] == path_single).all(), (
            f"Batch item {b} differs when run alone vs. in a batch"
        )