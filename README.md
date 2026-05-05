# kokoro-deutsch

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

```bash
git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch
cd kokoro-deutsch
uv sync
```

## German Frontend

This repository also contains `kokoro_de`, a small German frontend layer for Kokoro.

- Adds a repository-local German normalization layer for dates, times, euro amounts, percentages, selected units, ordinals, and common TTS/ML acronyms
- Keeps `espeak-de` as the default path and applies overrides only where needed
- Improves span-based matching for real-world spellings such as multi-word names and symbol forms
- Provides unit tests for the frontend layer

Example:

```python
from kokoro_de import phonemize

print(phonemize("Louis Vuitton auf Disney+"))
```

```python
from kokoro_de.pipeline import KokoroDEPipeline

pipeline = KokoroDEPipeline(lang_code="de", model=model)
for result in pipeline("Prime Video läuft auf dem iPad.", voice="df_voice"):
    audio = result.audio
```

Frontend tests:

```bash
python3 -m unittest discover -s tests
```

`KokoroDEPipeline` requires upstream `kokoro`. In a normal install it uses the published `kokoro` package; in a source checkout it can also fall back to the local `kokoro/` submodule.

## Repository Layout

```text
kokoro/              # Kokoro fork submodule (contains the `kokoro/` Python package)
StyleTTS2/           # Patched training code (git submodule: semidark/StyleTTS2)
scripts/             # Dataset prep, voicepack extraction, inference testing
configs/             # Training config(s)
docs/                # Training guide, troubleshooting, architecture notes
training/            # Local training artifacts metadata (audio excluded)
kokoro_de/           # German frontend package
tests/               # Frontend unit tests
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
