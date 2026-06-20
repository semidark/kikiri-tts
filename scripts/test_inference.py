#!/usr/bin/env python3
"""
Kokoro German: Test Inference
==============================
Tests the fine-tuned Kokoro model with a German phonetic test set.

Usage:
    # Zero-config sanity check (downloads reference model + voicepack from HF)
    python scripts/test_inference.py

    # Convert checkpoint + run inference
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --voicepack voices/dm_daniel_epoch3.pt \
        --output-dir test_output/epoch3

    # Use a previously converted model
    python scripts/test_inference.py \
        --model voices/kokoro_german_epoch3.pth \
        --voicepack voices/dm_daniel_epoch3.pt

    # Run on CPU
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --voicepack voices/dm_daniel_epoch3.pt \
        --device cpu
"""

import argparse
import sys
from pathlib import Path

# Prefer the kokoro submodule over any pip-installed kokoro package
_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

# Default reference model used for zero-config verification runs.
# When neither --checkpoint/--model nor --voicepack is provided, the script
# lazily downloads these from HuggingFace into a local cache directory so a
# fresh clone can run `uv run scripts/test_inference.py` with no arguments.
DEFAULT_REPO_ID = "kikiri-tts/kikiri-german-martin"
DEFAULT_MODEL_FILENAME = "kikiri_german_martin_ep10.pth"
DEFAULT_VOICE_FILENAME = "voices/martin.pt"
MODEL_CACHE_DIR = "test_output/.model_cache"


def download_reference_file(filename: str, cache_dir: str = MODEL_CACHE_DIR) -> str:
    """Lazily download a reference file from the default HF repo into a cache.

    Returns the local path. If the file is already cached, no download occurs.
    """
    from huggingface_hub import hf_hub_download

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    resolved = cache / filename
    if resolved.exists():
        print(f"Using cached reference file: {resolved}")
    else:
        print(f"Downloading reference file from {DEFAULT_REPO_ID}: {filename}...")
        hf_hub_download(
            repo_id=DEFAULT_REPO_ID,
            filename=filename,
            local_dir=str(cache),
        )
    return str(resolved)

# Standard German phonetic test set — covers all major pronunciation challenges
TEST_SENTENCES = [
    # 1. Umlauts (ä, ö, ü) and sch
    "Schön, dass du da bist. Die Bücher liegen auf dem großen Tisch.",
    # 2. Ich-Laut vs Ach-Laut (ç vs x)
    "Ich mache mich auf den Weg nach Aachen, um auch nachts wach zu sein.",
    # 3. Eszett (ß) and vowel length
    "Er aß die Maße in der Straße, aber das Maß war voll.",
    # 4. Zischlaute (z, ts) and consonant clusters
    "Zwei weiße Zwerge zwängen sich zwischen zwei Zweige.",
    # 5. Pf-Laute
    "Ein Pfau pflegt seine Federn an der Pfütze.",
    # 6. Prosody: questions and exclamations
    "Warum hast du das getan? Das ist ja unglaublich!",
    # 7. Numbers
    "Das kostet genau einhundertdreiundzwanzig Millionen Euro.",
]

# Pronunciation override test set — covers brand and technical term overrides
OVERRIDE_TEST_CASES = [
    ("Ich nutze GitHub und PyTorch.", ["github", "pytorch"]),
    ("Lade die API oder ein JSON herunter.", ["api", "json"]),
    ("Aktiviere CUDA auf deiner GPU.", ["cuda", "gpu"]),
    ("Nutze Claude oder HuggingFace.", ["claude", "huggingface"]),
    ("Füge einen Bindestrich in Disney+ oder espeak-ng ein.", ["disneyplus", "espeak-ng"]),
    ("Die Symbiose und Synthese sind biologische Begriffe.", ["symbiose", "synthese"]),
    ("Erstelle ein Zero-Shot Modell für maschinelles Lernen.", ["zero-shot"]),
]


def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the 5 inference components (bert, bert_encoder, predictor,
    text_encoder, decoder) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.

    Requires that training was done with the new parametrizations API
    (torch.nn.utils.parametrizations.weight_norm/spectral_norm) so the
    state dict keys are natively compatible with Kokoro's KModel.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have 'module.' prefix for KModel compatibility."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved Kokoro-format weights: {output} ({size_mb:.1f} MB)")
    return str(output)


def check_frontend():
    """Verify that German G2P overrides are loaded and correctly matched by the frontend pipeline.

    This operates entirely on phonemes and requires no model or voicepack.
    """
    try:
        from misaki.de import DEG2P, override_for
    except ImportError:
        print("ERROR: Could not import 'misaki.de'. Ensure misaki[de] is installed.")
        sys.exit(1)

    print("\n=== Running Frontend G2P Override Checks ===")
    g2p = DEG2P()
    failures = 0

    for i, (sentence, keys) in enumerate(OVERRIDE_TEST_CASES):
        # DEG2P returns (phonemes, tokens)
        phonemes, _ = g2p(sentence)
        print(f"[{i + 1}/{len(OVERRIDE_TEST_CASES)}] Text: '{sentence}'")
        print(f"      Phonemes: '{phonemes}'")

        case_failed = False
        for key in keys:
            expected_ipa = override_for(key)
            if not expected_ipa:
                print(f"  ❌ FAIL: Key '{key}' has no defined override in misaki.")
                case_failed = True
                continue

            clean_phonemes = phonemes.replace(" ", "")
            clean_expected = expected_ipa.replace(" ", "")
            if clean_expected not in clean_phonemes:
                print(f"  ❌ FAIL: Expected override '{key}' ('{expected_ipa}') not found in phonemes.")
                case_failed = True
            else:
                print(f"  ✅ PASS: Found '{key}' ('{expected_ipa}')")

        if case_failed:
            failures += 1

    print("=" * 44)
    if failures == 0:
        print("🎉 ALL FRONTEND OVERRIDE CHECKS PASSED!\n")
        sys.exit(0)
    else:
        print(f"❌ {failures} FRONTEND OVERRIDE CHECKS FAILED.\n")
        sys.exit(1)


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    device: str = "auto",
):
    """Run inference on the German test set."""
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with our fine-tuned weights and config
    print(f"Loading model from: {model_path}")
    print(f"  Config: {config_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = kmodel.to(device).eval()

    # Create pipeline with German lang_code
    pipeline = KPipeline(lang_code="d", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    # Load voicepack
    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate audio for each test sentence
    print(f"\nGenerating {len(TEST_SENTENCES)} phonetic test sentences...\n")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"[{i + 1}/{len(TEST_SENTENCES)}] {text[:60]}...")
        try:
            generator = pipeline(text, voice=voice, speed=1)
            all_audio = []
            for gs, ps, audio in generator:
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                import numpy as np

                combined = np.concatenate(all_audio)
                wav_path = out / f"test_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print(f"  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nGenerating {len(OVERRIDE_TEST_CASES)} override test sentences...\n")
    for i, (text, _) in enumerate(OVERRIDE_TEST_CASES):
        print(f"[{i + 1}/{len(OVERRIDE_TEST_CASES)}] {text[:60]}...")
        try:
            generator = pipeline(text, voice=voice, speed=1)
            all_audio = []
            for gs, ps, audio in generator:
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                import numpy as np

                combined = np.concatenate(all_audio)
                wav_path = out / f"override_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print(f"  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Test audio saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Kokoro German model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--checkpoint",
        help="Path to StyleTTS2 checkpoint (.pth) — will be converted automatically",
    )
    group.add_argument(
        "--model",
        help="Path to already-converted Kokoro-format weights (.pth). "
        f"If omitted (and no --checkpoint), downloads '{DEFAULT_MODEL_FILENAME}' "
        f"from {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--voicepack",
        required=False,
        help="Path to voicepack (.pt). If omitted, downloads "
        f"'{DEFAULT_VOICE_FILENAME}' from {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--config",
        default="training/config.json",
        help="Path to Kokoro config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output/",
        help="Directory to save generated WAV files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--check-frontend",
        action="store_true",
        help="Only run fast G2P frontend override verification checks, without generating audio",
    )

    args = parser.parse_args()

    if args.check_frontend:
        check_frontend()
        return

    # Resolve model path: convert checkpoint, use explicit model, or
    # fall back to the default reference model from HuggingFace.
    if args.checkpoint:
        model_path = convert_checkpoint(
            args.checkpoint,
            str(Path(args.output_dir) / "kokoro_german_converted.pth"),
        )
    elif args.model:
        model_path = args.model
    else:
        model_path = download_reference_file(DEFAULT_MODEL_FILENAME)

    # Resolve voicepack path: use explicit voicepack or fall back to the
    # default reference voicepack from HuggingFace.
    if args.voicepack:
        voicepack_path = args.voicepack
    else:
        voicepack_path = download_reference_file(DEFAULT_VOICE_FILENAME)

    run_inference(
        model_path=model_path,
        voicepack_path=voicepack_path,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
