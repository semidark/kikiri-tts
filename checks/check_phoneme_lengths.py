import os
from pathlib import Path

import pytest
import yaml

from text_utils import TextCleaner


LIMIT = 512
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config_german_ft.yml"


def _load_target_files():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data_params"]
    return [
        str(_PROJECT_ROOT / data["train_data"]),
        str(_PROJECT_ROOT / data["val_data"]),
        str(_PROJECT_ROOT / data["OOD_data"]),
    ]


TARGET_FILES = _load_target_files()


def collect_violations(file_path, limit=LIMIT):
    """Return list of (line_number, token_length, path_field) for lines that exceed limit."""
    cleaner = TextCleaner()
    violations = []
    with open(file_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            length = len(cleaner(parts[1]))
            if length > limit:
                violations.append((line_number, length, parts[0]))
    return violations


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_path", TARGET_FILES)
def test_no_phoneme_sequence_exceeds_bert_limit(file_path):
    if not os.path.exists(file_path):
        pytest.skip(f"Datei nicht gefunden: {file_path}")

    violations = collect_violations(file_path)

    if violations:
        lines = "\n".join(
            f"  Zeile {ln}: {length} Tokens — {path}"
            for ln, length, path in violations
        )
        pytest.fail(
            f"{len(violations)} Zeile(n) in {file_path} überschreiten das BERT-Limit ({LIMIT}):\n{lines}"
        )


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for file_path in TARGET_FILES:
        if not os.path.exists(file_path):
            print(f"[-] Datei nicht gefunden: {file_path}")
            continue
        print(f"[*] Prüfe Datei: {file_path}")
        violations = collect_violations(file_path)
        total = sum(1 for _ in open(file_path, encoding="utf-8"))
        if violations:
            for ln, length, path in violations:
                print(f"    ! Zeile {ln}: {length} Tokens (Limit: {LIMIT})")
                print(f"      Pfad: {path}")
            print(f"[-] KRITISCH: {len(violations)} Zeile(n) werden BERT zum Absturz bringen!\n")
        else:
            print(f"[+] Erfolg: Alle {total} Zeilen sind sicher ({LIMIT} Tokens).\n")