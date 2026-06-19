# Training Guide

This guide is the practical path for fine-tuning Kokoro-82M for German.

For deep debugging details, see `TROUBLESHOOTING.md`.
For architecture and compatibility notes, see `ARCHITECTURE.md`.

## 1) Prerequisites

### Hardware

| Hardware | Status | Notes |
|----------|--------|-------|
| NVIDIA (CUDA) | Recommended | Any GPU with 10GB+ VRAM. batch_size=4 works on 12GB. |
| AMD (ROCm) | Works with caveats | See [AMD ROCm Notes](TROUBLESHOOTING.md#amd-rocm-notes). |
| CPU only | Not practical | Training would take weeks. |

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng libsndfile1

# macOS
brew install espeak-ng libsndfile
```

Both are mandatory. `espeak-ng` drives the IPA phonemization (`misaki`). `libsndfile` is required by `soundfile` for WAV I/O.

### Python Environment

- Python: **3.10–3.12** (the `spacy` dependency only ships wheels up to cp312).
  The repo pins `3.12` in `.python-version` so `uv` picks a working interpreter.
- Package manager: `uv`

This repo uses **git submodules** (`kokoro/` and `StyleTTS2/`). Clone with
`--recurse-submodules` or those directories will be empty:

```bash
git clone --recurse-submodules https://github.com/semidark/kikiri-tts
cd kikiri-tts
uv sync
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

The training environment for StyleTTS2 is separate and uses a venv:

```bash
pip install torch torchaudio  # match your CUDA/ROCm version
pip install accelerate transformers
pip install librosa soundfile pyyaml tensorboard
pip install munch phonemizer huggingface_hub
pip install Cython  # required to build monotonic_align
```

## 2) Prepare Dataset

Create file lists in StyleTTS2 format:

`path/to/audio.wav|IPA phoneme string|speaker_name`

Requirements:
- WAV, mono, 24kHz, 16-bit
- Typical clip duration: 2–30s
- Keep phoneme strings compatible with Kokoro symbols (see [Phoneme Compatibility](ARCHITECTURE.md#german-phoneme-compatibility))

German G2P example:

```python
from misaki import espeak

g2p = espeak.EspeakG2P(language='de')
phonemes, _ = g2p(text)
phonemes = phonemes.replace('ʏ', 'y')  # ʏ not in Kokoro vocab
```

Use:
- `scripts/prepare_dataset.py`
- `scripts/prepare_training.py`

## 3) Prepare Base Weights

Convert Kokoro HuggingFace weights into StyleTTS2-compatible checkpoint format:

```python
import torch

raw = torch.load('kokoro-v1_0.pth', weights_only=False)

def strip_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

net = {
    'bert': strip_prefix(raw['bert']),
    'bert_encoder': strip_prefix(raw['bert_encoder']),
    'predictor': strip_prefix(raw['predictor']),
    'text_encoder': strip_prefix(raw['text_encoder']),
    'decoder': strip_prefix(raw['decoder']),
}
torch.save({'net': net}, 'kokoro_base.pth')
```

Set `load_only_params: true` in the config so StyleTTS2 uses `strict=False` when loading — this silently ignores missing keys for components not present in Kokoro (diffusion network, SLM discriminator).

## 4) Symbol Mapping (Critical)

StyleTTS2 default token indices do not match Kokoro token indices.

Required:
- `StyleTTS2/text_utils.py` must import from `kokoro_symbols.py`
- `kokoro_symbols.py` must contain the 178-token Kokoro mapping

Verify:

```python
from kokoro_symbols import symbols, dicts, TextCleaner
assert len(symbols) == 178
tc = TextCleaner()
assert dicts['ç'] == 78   # ich-Laut
assert dicts['ʦ'] == 20   # ts affricate
assert dicts['ː'] == 158  # length mark
```

Without this, training appears to run but token embeddings are silently wrong.

## 5) StyleTTS2 Environment

`StyleTTS2/` is a patched submodule with required fixes already included.

You still need utility models:
- `Utils/JDC/bst.t7`
- `Utils/ASR/config.yml` and `Utils/ASR/epoch_00080.pth`
- `Utils/PLBERT/*`

Build monotonic alignment extension:

```bash
cd StyleTTS2
git clone https://github.com/resemble-ai/monotonic_align.git
cd monotonic_align
python setup.py build_ext --inplace
```

## 6) Configure Training

Primary config: `configs/config_german_ft.yml`

### Critical: Top-Level vs Nested Parameters

`train_first.py` reads critical parameters from the **top level** of the YAML, not from the nested `training:` block:

```python
# These read from TOP LEVEL — not from training:
batch_size = config.get("batch_size", 10)
epochs = config.get("epochs_1st", 200)
saving_epoch = config.get("save_freq", 2)
pretrained_model = config["pretrained_model"]
load_only_params = config.get("load_only_params", True)
```

If you put these only inside `training:`, they will be silently ignored and defaults will be used. Always set them at the top level.

### Important Stage 2 Settings

```yaml
second_stage_load_pretrained: false  # load from first_stage.pth (recommended)
joint_epoch: 3                       # start adversarial training at epoch 3
lambda_slm: 1.0                      # enable SLM adversarial loss
```

See `TROUBLESHOOTING.md` for why these values matter.

## 7) Smoke Test

Before long runs, verify each component:

1. **Symbol map:** loads and has length 178 (see step 4)
2. **Model loads:** no size mismatch errors (missing keys for diffusion/SLM are expected)
3. **Forward + backward pass:** all losses are finite (not NaN or inf)
4. **Run 2 training steps** and Ctrl+C after confirming non-NaN losses

Healthy first-step losses (approximate):

| Loss | Expected range |
|------|---------------|
| Mel Loss | 0.8–1.5 (drops fast) |
| Gen Loss | 3–6 |
| Disc Loss | 4–6 |
| Mono Loss | 0.01–0.1 |
| S2S Loss | 1–6 (drops over epochs) |
| SLM Loss | 1–3 |

Any NaN in Mel Loss is a red flag — most likely a symbol mapping problem.

## 8) Stage 1 Training

Run from `StyleTTS2/`:

```bash
accelerate launch train_first.py --config_path ../configs/config_german_ft.yml
```

### Loss interpretation

| Loss | What it means | Healthy trend |
|------|--------------|---------------|
| Mel Loss | Mel spectrogram reconstruction | 0.8 → 0.25 over 10 epochs |
| Gen Loss | GAN generator vs discriminator | Stable 2.5–3.5 |
| Disc Loss | GAN discriminator | Stable 3.8–4.2 |
| Mono Loss | Monotonic alignment quality | < 0.05 — lower is better |
| S2S Loss | Sequence-to-sequence alignment | Declining over time |
| SLM Loss | WavLM feature matching | Stable or slowly declining |

If Mel Loss plateaus above 0.4 after several epochs, check data quality, phoneme mapping, or learning rate.

Checkpoints saved in `StyleTTS2/logs/<run>/`.

![Stage 1 TensorBoard](images/tensorboard_stage1.png)

## 9) Stage 2 Training

Run from `StyleTTS2/`:

```bash
accelerate launch train_second.py --config_path ../configs/config_german_ft.yml
```

### Loss interpretation

| Loss | What it means | Healthy trend |
|------|--------------|---------------|
| Mel Loss | Mel reconstruction with predicted prosody | ~0.43 at start, declining to ~0.25 |
| Dur Loss | Duration prediction accuracy | 1.3 → 0.9 over 10 epochs |
| CE Loss | Alignment cross-entropy | 0.18 → 0.05 |
| Norm Loss | Energy prediction | 3.0 → 0.8 (noisy) |
| F0 Loss | Pitch contour prediction | 4.1 → 1.8 over 10 epochs |
| Gen/Disc Loss | GAN discriminator losses | Activate at `joint_epoch`, stable ~2–4 |
| SLM Loss | WavLM adversarial loss | Activate at `joint_epoch`, stable ~1–3 |

**Key indicator:** Stage 2 Mel loss should start at **~0.43** (pretrained weights loaded correctly), not **~7.5–8.0** (random initialization). If you see Mel loss starting above 2.0, see `TROUBLESHOOTING.md`.

![Stage 2 TensorBoard](images/tensorboard_stage2-try2.png)

## 10) Extract Voicepack and Test Inference

Extract:

```bash
python scripts/extract_voicepack.py \
  --model StyleTTS2/logs/kokoro-deutsch/epoch_2nd_00009.pth \
  --audio-dir path/to/audio \
  --output voices/dm_daniel.pt
```

Convert/test inference:

```bash
python scripts/test_inference.py \
  --checkpoint StyleTTS2/logs/kokoro-deutsch/epoch_2nd_00009.pth \
  --voicepack voices/dm_daniel.pt \
  --output-dir test_output/
```

## 11) Quick Checklist

- [ ] System dependencies installed (espeak-ng, libsndfile)
- [ ] Dataset lists are valid
- [ ] Symbol mapping is Kokoro-compatible (178 tokens)
- [ ] Base weights converted
- [ ] Utility models downloaded
- [ ] Config uses top-level keys (not nested `training:` block)
- [ ] Smoke test passes with finite losses
- [ ] Stage 1 completes and checkpoints save
- [ ] Stage 2 starts from trained weights (Mel ~0.43, not ~7.5)
- [ ] Voicepack extraction succeeds
- [ ] Inference audio is intelligible