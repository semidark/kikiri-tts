#!/usr/bin/env python3
"""
Triage overrides into keep, review, and drop-for-now.

Heuristics:
- keep obvious project-critical brand/model names
- review ambiguous brand names and nearly all DE_FOREIGN items
- drop EN overrides that are not currently relevant to this repository corpus
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kokoro_de import BRAND_OVERRIDES, DE_FOREIGN, EN_OVERRIDES, normalize_for_lookup


KEEP_BRAND = {
    "claude",
    "coqui",
    "cuda",
    "deepseek",
    "espeak-ng",
    "fastpitch",
    "geforce",
    "gemini",
    "hifigan",
    "intellij",
    "kikiri",
    "kokoro",
    "llama",
    "mistral",
    "mixtral",
    "neovim",
    "phonemizer",
    "piper",
    "pycharm",
    "qwen",
    "radeon",
    "rocm",
    "ryzen",
    "tacotron",
    "tacotron2",
    "typescript",
    "unsloth",
    "vscode",
}

REVIEW_BRAND = {
    "bark",
    "matcha",
    "triton",
    "vits",
}

KEEP_FOREIGN = {
    "synthese",
}

REVIEW_EN = {
    "accelerate",
    "amd",
    "apache",
    "api",
    "bert",
    "checkpoint",
    "cli",
    "cpu",
    "debian",
    "disneyplus",
    "dropout",
    "fallback",
    "finetuning",
    "gan",
    "github",
    "githubactions",
    "gpu",
    "https",
    "huggingface",
    "ipad",
    "jameswebb",
    "json",
    "kde",
    "louisvuitton",
    "macos",
    "moetchandon",
    "nvidia",
    "ollama",
    "pipeline",
    "primevideo",
    "protocol",
    "pytorch",
    "rag",
    "repository",
    "review",
    "rnn",
    "rocm",
    "routing",
    "runtime",
    "styletts",
    "styletts2",
    "surface",
    "tcp",
    "thread",
    "tpu",
    "training",
    "transformers",
    "ui",
    "ubuntu",
    "wavlm",
    "whisper",
    "wsl",
    "zero-shot",
}


def repo_corpus() -> str:
    parts: list[str] = []
    include_roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "PROGRESS.md",
        REPO_ROOT / "docs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "configs",
        REPO_ROOT / "training",
        REPO_ROOT / "kokoro_de",
    ]
    for root in include_roots:
        if not root.exists():
            continue
        if root.is_file():
            parts.append(root.read_text(encoding="utf-8", errors="ignore"))
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "overrides.py":
                continue
            if path.suffix.lower() not in {".md", ".py", ".txt", ".json", ".yml", ".yaml"}:
                continue
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def normalized_repo_index(corpus: str) -> set[str]:
    raw_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9+/#&._-]+", corpus)
    normalized: set[str] = set()
    for size in (1, 2, 3):
        for i in range(len(raw_tokens) - size + 1):
            chunk = " ".join(raw_tokens[i : i + size])
            key = normalize_for_lookup(chunk)
            if key:
                normalized.add(key)
    return normalized


def decide_brand(key: str) -> tuple[str, str]:
    if key in KEEP_BRAND:
        return "keep", "project-critical tech/model/tts term"
    if key in REVIEW_BRAND:
        return "review", "ambiguous common noun or acronym despite likely relevance"
    return "review", "brand override not explicitly curated yet"


def decide_foreign(key: str) -> tuple[str, str]:
    if key in KEEP_FOREIGN:
        return "keep", "user-confirmed missing from baseline"
    return "review", "german foreign-word stress candidate needs listening check"


def decide_en(key: str) -> tuple[str, str]:
    if key in REVIEW_EN:
        return "review", "explicitly relevant technical/repo term"
    return "drop", "not currently relevant to repository corpus"


def main() -> None:
    output_dir = REPO_ROOT / "artifacts" / "override_triage"
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = repo_corpus()
    _ = normalized_repo_index(corpus)

    rows: list[dict] = []
    review_keys: list[str] = []
    keep_keys: list[str] = []
    drop_keys: list[str] = []

    for category, mapping in [
        ("brand", BRAND_OVERRIDES),
        ("foreign", DE_FOREIGN),
        ("en", EN_OVERRIDES),
    ]:
        for key in sorted(mapping):
            if category == "brand":
                decision, reason = decide_brand(key)
            elif category == "foreign":
                decision, reason = decide_foreign(key)
            else:
                decision, reason = decide_en(key)
            row = {
                "category": category,
                "key": key,
                "decision": decision,
                "reason": reason,
            }
            rows.append(row)
            if decision == "review":
                review_keys.append(key)
            elif decision == "keep":
                keep_keys.append(key)
            else:
                drop_keys.append(key)

    triage_path = output_dir / "triage.jsonl"
    with triage_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "review_keys.txt").write_text("\n".join(review_keys) + "\n", encoding="utf-8")
    (output_dir / "keep_keys.txt").write_text("\n".join(keep_keys) + "\n", encoding="utf-8")
    (output_dir / "drop_keys.txt").write_text("\n".join(drop_keys) + "\n", encoding="utf-8")
    summary = {
        "keep": len(keep_keys),
        "review": len(review_keys),
        "drop": len(drop_keys),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
