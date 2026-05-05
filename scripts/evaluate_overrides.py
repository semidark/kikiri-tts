#!/usr/bin/env python3
"""
Evaluate kokoro_de overrides against plain espeak-de and optionally render A/B WAVs.

This script is intentionally self-contained:
- downloads the published Victoria model + voicepack into a project-local HF cache
- avoids importing Kokoro's KPipeline, which would pull in English G2P extras
- compares plain espeak-de output against kokoro_de's routed output
- can render A/B clips for human listening
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from misaki import espeak


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kokoro_de import BRAND_OVERRIDES, DE_FOREIGN, EN_OVERRIDES, Router


HF_REPO_ID = "kikiri-tts/kikiri-german-victoria"
HF_MODEL_FILE = "kikiri_german_victoria_ep10.pth"
HF_VOICE_FILE = "voices/victoria.pt"

SURFACE_ALIASES = {
    "disneyplus": "Disney+",
    "githubactions": "GitHub Actions",
    "jameswebb": "James Webb",
    "louisvuitton": "Louis Vuitton",
    "moetchandon": "Moët & Chandon",
    "primevideo": "Prime Video",
    "twitterx": "Twitter/X",
    "veuveclicquot": "Veuve Clicquot",
    "zero-shot": "Zero-Shot",
}

CATEGORIES = {
    "brand": BRAND_OVERRIDES,
    "en": EN_OVERRIDES,
    "foreign": DE_FOREIGN,
}


@dataclass(frozen=True)
class EvalCase:
    category: str
    key: str
    surface: str
    override_phonemes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate kokoro_de overrides.")
    parser.add_argument(
        "--categories",
        default="brand,en,foreign",
        help="Comma-separated categories: brand,en,foreign",
    )
    parser.add_argument(
        "--mode",
        choices=["scan", "render", "both"],
        default="both",
        help="Run metadata scan only, audio render only, or both.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/override_eval",
        help="Output directory inside the project.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit after filtering. 0 means all matching cases.",
    )
    parser.add_argument(
        "--keys-file",
        default="",
        help="Optional text file with one override key per line to restrict evaluation.",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only keep cases where override phonemes differ from plain espeak-de.",
    )
    parser.add_argument(
        "--carrier-template",
        default="{text}",
        help="Template used for rendering. Example: 'Das Wort lautet: {text}.'",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="WAV sample rate.",
    )
    return parser.parse_args()


def _load_kmodel_class():
    pkg_path = REPO_ROOT / "kokoro" / "kokoro"
    pkg = types.ModuleType("kokoro")
    pkg.__path__ = [str(pkg_path)]
    sys.modules["kokoro"] = pkg
    spec = importlib.util.spec_from_file_location("kokoro.model", pkg_path / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load kokoro.model from local submodule")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kokoro.model"] = mod
    spec.loader.exec_module(mod)
    return mod.KModel


def _prepare_hf_cache() -> Path:
    cache_dir = REPO_ROOT / "artifacts" / "hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    return cache_dir


def resolve_victoria_assets() -> tuple[Path, Path]:
    _prepare_hf_cache()
    model_path = Path(hf_hub_download(HF_REPO_ID, filename=HF_MODEL_FILE))
    voice_path = Path(hf_hub_download(HF_REPO_ID, filename=HF_VOICE_FILE))
    return model_path, voice_path


def build_cases(category_names: Iterable[str], selected_keys: set[str] | None = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for category in category_names:
        mapping = CATEGORIES[category]
        for key, override_phonemes in sorted(mapping.items()):
            if selected_keys is not None and key not in selected_keys:
                continue
            surface = SURFACE_ALIASES.get(key, key)
            cases.append(
                EvalCase(
                    category=category,
                    key=key,
                    surface=surface,
                    override_phonemes=override_phonemes,
                )
            )
    return cases


def scan_cases(cases: list[EvalCase]) -> list[dict]:
    plain_g2p = espeak.EspeakG2P(language="de")
    router = Router(g2p=plain_g2p)
    rows: list[dict] = []
    for case in cases:
        plain_phonemes = plain_g2p(case.surface)
        if isinstance(plain_phonemes, tuple):
            plain_phonemes = plain_phonemes[0] or ""
        override_phonemes = router.phonemize(case.surface)
        rows.append(
            {
                "category": case.category,
                "key": case.key,
                "surface": case.surface,
                "plain_phonemes": plain_phonemes,
                "override_phonemes": override_phonemes,
                "changed": plain_phonemes != override_phonemes,
            }
        )
    return rows


def filter_rows(rows: list[dict], changed_only: bool, limit: int) -> list[dict]:
    if changed_only:
        rows = [row for row in rows if row["changed"]]
    if limit > 0:
        rows = rows[:limit]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def slugify(text: str) -> str:
    chars = []
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


def render_audio(
    rows: list[dict],
    output_dir: Path,
    carrier_template: str,
    device_name: str,
    sample_rate: int,
) -> list[dict]:
    model_path, voice_path = resolve_victoria_assets()
    KModel = _load_kmodel_class()
    model = KModel(repo_id="hexgrad/Kokoro-82M", model=str(model_path)).to(device_name).eval()
    voice = torch.load(voice_path, map_location="cpu", weights_only=True)
    plain_g2p = espeak.EspeakG2P(language="de")
    router = Router(g2p=plain_g2p)

    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        prompt_text = carrier_template.format(text=row["surface"])
        plain_phonemes = plain_g2p(prompt_text)
        if isinstance(plain_phonemes, tuple):
            plain_phonemes = plain_phonemes[0] or ""
        override_phonemes = router.phonemize(prompt_text)
        plain_audio = model(plain_phonemes, voice[len(plain_phonemes) - 1], return_output=True).audio
        override_audio = model(override_phonemes, voice[len(override_phonemes) - 1], return_output=True).audio

        base_name = f"{index:04d}_{row['category']}_{slugify(row['surface'])}"
        plain_path = wav_dir / f"{base_name}_plain.wav"
        override_path = wav_dir / f"{base_name}_override.wav"
        sf.write(plain_path, plain_audio.numpy(), sample_rate)
        sf.write(override_path, override_audio.numpy(), sample_rate)

        rendered = dict(row)
        rendered.update(
            {
                "prompt_text": prompt_text,
                "plain_render_phonemes": plain_phonemes,
                "override_render_phonemes": override_phonemes,
                "plain_wav": str(plain_path.relative_to(REPO_ROOT)),
                "override_wav": str(override_path.relative_to(REPO_ROOT)),
                "plain_duration_s": round(len(plain_audio) / sample_rate, 3),
                "override_duration_s": round(len(override_audio) / sample_rate, 3),
            }
        )
        results.append(rendered)
    return results


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [name.strip() for name in args.categories.split(",") if name.strip()]
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {', '.join(unknown)}")

    selected_keys = None
    if args.keys_file:
        keys_path = REPO_ROOT / args.keys_file
        selected_keys = {
            line.strip()
            for line in keys_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    cases = build_cases(categories, selected_keys=selected_keys)
    scan_rows = scan_cases(cases)
    scan_manifest = output_dir / "scan_all.jsonl"
    write_jsonl(scan_manifest, scan_rows)

    filtered_rows = filter_rows(scan_rows, changed_only=args.changed_only, limit=args.limit)
    filtered_manifest = output_dir / "scan_selected.jsonl"
    write_jsonl(filtered_manifest, filtered_rows)

    summary = {
        "total_cases": len(scan_rows),
        "selected_cases": len(filtered_rows),
        "changed_cases": sum(1 for row in scan_rows if row["changed"]),
        "categories": categories,
        "changed_only": args.changed_only,
        "limit": args.limit,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.mode in {"render", "both"}:
        device_name = choose_device(args.device)
        rendered_rows = render_audio(
            rows=filtered_rows,
            output_dir=output_dir,
            carrier_template=args.carrier_template,
            device_name=device_name,
            sample_rate=args.sample_rate,
        )
        render_manifest = output_dir / "rendered.jsonl"
        write_jsonl(render_manifest, rendered_rows)
        print(f"Rendered {len(rendered_rows)} A/B pairs to {output_dir}")


if __name__ == "__main__":
    main()
