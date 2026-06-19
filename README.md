# kikiri-tts

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

## Published Model

### German Multi-Speaker Base Model (Stage 1)

**dida-80b/kokoro-deutsch-hui-base** is available on HuggingFace.

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

This is a base model, not a finished single-speaker voice.

## Quick Setup

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng libsndfile1

# macOS
brew install espeak-ng libsndfile
```

### Clone

> [!IMPORTANT]
> The `kokoro/` and `StyleTTS2/` code lives in **git submodules**. Clone with
> `--recurse-submodules`, or run `git submodule update --init --recursive` if
> you already cloned without them — otherwise those directories will be empty.

```bash
git clone --recurse-submodules https://github.com/semidark/kikiri-tts
cd kikiri-tts
uv sync
```

`uv sync` selects a compatible interpreter via the pinned `.python-version`
(Python 3.12; see [Python version](docs/TRAINING_GUIDE.md#python-environment)).

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
