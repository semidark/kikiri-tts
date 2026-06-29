# kikiri-tts

<img src="docs/images/kikiri-tts-logo.png" alt="kikiri-tts logo" width="150">

> [!NOTE]
> This repository was formerly named `kokoro-deutsch`. The Python package and the
> published HuggingFace model still use the old `kokoro-deutsch` name.

Training recipe for fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) for German with a patched [StyleTTS2](https://github.com/yl4579/StyleTTS2) submodule.

## What This Is

- A reproducible fine-tuning workflow (dataset prep -> Stage 1 -> Stage 2 -> voicepack extraction)
- Original scripts for data preparation and checkpoint/voicepack conversion
- A patched `StyleTTS2/` submodule with the fixes required for stable Stage 2 training

## What This Is Not

- Not a general-purpose Kokoro replacement repository
- Not a bundled upstream mirror of `demo/`, `examples/`, `kokoro.js/`, or `tests/`
- Not a redistributable training dataset

## Start Here

### I want to train my own German voice

Start with `docs/TRAINING_GUIDE.md`.

### I am debugging training failures

Go to `docs/TROUBLESHOOTING.md`.

### I want architecture details and compatibility notes

See `docs/ARCHITECTURE.md`.

## Status

The end-to-end pipeline is working:

`Dataset preparation -> Weight conversion -> Stage 1 -> Stage 2 -> Voicepack extraction -> KModel inference`

## Published Models & Voices

All checkpoints are compatible with the Kokoro-82M inference pipeline.

### Base Model

**dida-80b/kokoro-deutsch-hui-base** is a German multi-speaker Stage 1 base model
trained on ~51 hours of audio. It is not a finished single-speaker voice —
use it as a starting point for training your own with
`docs/TRAINING_GUIDE.md`.

| Specification | Value |
|---|---|
| Speakers | 51 (24M / 27F) |
| Training Audio | ~51 hours (effective) |
| Train Samples | 20,495 |
| Val Samples | 418 |
| Final Mel Loss | 0.3264 |
| License | CC0-1.0 |
| Model | [dida-80b/kokoro-deutsch-hui-base](https://huggingface.co/dida-80b/kokoro-deutsch-hui-base) |
| Dataset | [dida-80b/hui-german-51speakers](https://huggingface.co/datasets/dida-80b/hui-german-51speakers) |

### Fine-Tuned Voices

Stage 2 single-speaker fine-tunings. Each ships with a speaker voicepack.
Click the links to hear speech demos directly from HuggingFace.

#### German (de)

| Voice | Speaker | Samples | License | Demo |
|-------|---------|---------|---------|------|
| **[kikiri-german-martin](https://huggingface.co/kikiri-tts/kikiri-german-martin)** | Martin Harbecke (male) | 627 | Apache 2.0 | [Speech Demo](https://huggingface.co/kikiri-tts/kikiri-german-martin/blob/main/README.md#demo) |
| **[kikiri-german-victoria](https://huggingface.co/kikiri-tts/kikiri-german-victoria)** | Victoria Asztaller (female) | 455 | Apache 2.0 | [Speech Demo](https://huggingface.co/kikiri-tts/kikiri-german-victoria#demo) |

*More voices and languages coming soon. Check the [kikiri-tts](https://huggingface.co/kikiri-tts) HuggingFace organization for updates.*

## Running Verification Tests

To run a quick text-to-speech sanity check — e.g. after updating phonemizer
packages like `misaki`, bumping dependencies, or making model changes — run the
inference script with no arguments:

```bash
uv run scripts/test_inference.py
```

On first run this downloads a reference model
([`kikiri-tts/kikiri-german-martin`](https://huggingface.co/kikiri-tts/kikiri-german-martin))
and voicepack into a local cache (`test_output/.model_cache/`), synthesizes the
standard German phonetic test sentences, and writes the audio to `test_output/`.
It runs on CPU automatically when no GPU is available, so it works on any machine
and in CI/CD pipelines. Subsequent runs reuse the cached files.

To test a specific checkpoint or voice instead, pass explicit paths:

```bash
uv run scripts/test_inference.py \
    --model voices/kokoro_german_epoch3.pth \
    --voicepack voices/dm_daniel_epoch3.pt \
    --device cpu
```

## Repository Layout

```text
kokoro/              # Kokoro fork submodule (contains the `kokoro/` Python package)
StyleTTS2/           # Patched training code (git submodule: semidark/StyleTTS2)
scripts/             # Dataset prep, voicepack extraction, inference testing
configs/             # Training config(s)
docs/                # Training guide, troubleshooting, architecture notes
training/            # Local training artifacts metadata (audio excluded)
```

## Contributing

Contributions are welcome, especially:

- Reproducible runs on public datasets
- Fine-tuning recipes for other languages
- Training stability and quality improvements

## Attribution

See `NOTICE` for upstream attribution and license details.

## License

Apache License 2.0 — see `LICENSE`.
