# Troubleshooting

This document collects the critical training failures found during German Kokoro fine-tuning and how they were fixed.

If you are looking for the step-by-step path, use `TRAINING_GUIDE.md`.

## Setup / Environment

### espeak data error: `Error processing file '.../phontab': No such file or directory`

If German G2P fails on import or first synthesis with an error pointing at a
`/home/runner/work/.../espeak-ng-data/phontab` path, the bundled
`espeakng-loader` wheel is at fault: its prebuilt `libespeak-ng.so` has a
CI build path compiled in and ignores the data path it is handed (see
[espeakng-loader#5](https://github.com/thewh1teagle/espeakng-loader/issues/5)).

Fix: install the system `espeak-ng` (already required) and point the bundled
loader at it. Replace the two bundled artifacts in your venv with symlinks to
the system install:

```bash
EL=$(uv run python -c 'import espeakng_loader, os; print(os.path.dirname(espeakng_loader.__file__))')
ln -sf /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 "$EL/libespeak-ng.so"
ln -sf /usr/lib/x86_64-linux-gnu/espeak-ng-data    "$EL/espeak-ng-data"
```

This lives inside `.venv`, so re-apply it after a fresh `uv sync`. Adjust the
paths for your platform (e.g. `aarch64-linux-gnu`, or the `brew` prefix on macOS).

## Stage 2 Static Noise / Collapsed Style Encoder

### Symptoms

- Stage 1 sounds usable, Stage 2 outputs static noise
- Voicepack norm collapses or explodes
- Stage 2 mel starts in a bad regime (acts like random init)

### Root Causes and Fixes

1) DataParallel wrap order was wrong in `train_second.py`
- Cause: wrapping model before checkpoint load changed key names (`module.` prefix)
- Fix: load checkpoint first, wrap after

2) `ignore_modules` excluded pretrained prosody modules
- Cause: `bert`, `bert_encoder`, `predictor` were dropped
- Fix: stop excluding those modules

3) Missing adversarial ground-truth tensors
- Cause: `y_rec_gt` / `y_rec_gt_pred` path had been removed
- Fix: restore computation before adversarial loss section

4) GAN discriminator gated on wrong epoch variable
- Cause: gated on `diff_epoch` instead of `joint_epoch`
- Fix: gate on `joint_epoch`

5) Diffusion sampler used even when diffusion was disabled
- Cause: invalid style embeddings fed to discriminator
- Fix: bypass diffusion sampling when diffusion is disabled

## Known Issues and Fixes

### 1) F0 shape mismatch at epoch boundaries

Files:
- `StyleTTS2/train_first.py`
- `StyleTTS2/train_second.py`

Fix:
- Remove the extra `unsqueeze(0)` on `F0_real` in TensorBoard audio generation path.

### 2) Checkpoint saved too late (after audio generation)

File:
- `StyleTTS2/train_first.py`

Fix:
- Save checkpoint before TensorBoard audio generation block.

### 3) Missing `.train()` restoration after checkpoint load

File:
- `StyleTTS2/train_second.py`

Fix:
- Restore train mode for all relevant modules, not only a subset.

### 4) Hardcoded debugger breakpoint

File:
- `StyleTTS2/train_second.py`

Fix:
- Remove `set_trace()` calls that hang under `accelerate launch`.

### 5) PLBERT max sequence overflow

File:
- `StyleTTS2/meldataset.py`

Fix:
- Filter samples whose cleaned token length exceeds 510.

### 6) PyTorch 2.6+ `torch.load` default changed

Files:
- `StyleTTS2/train_first.py`
- `StyleTTS2/train_second.py`

Fix:
- Ensure legacy checkpoints are loaded with `weights_only=False`.

## Validation Signals That Things Are Healthy

- Stage 2 starts from trained behavior, not random-init behavior
- Losses stay finite
- Voicepack norms remain stable and reasonable
- TensorBoard audio improves over epochs

## Recommended Debug Flow

1. Confirm symbol mapping compatibility first
2. Confirm checkpoint load paths and key matching
3. Confirm train/eval mode transitions
4. Confirm adversarial gating and diffusion bypass settings
5. Run a short smoke run before full training

## AMD ROCm Notes

### What works

- **ROCm 7.12** on AMD Radeon 8060S (Strix Halo / Ryzen AI Max) runs the full training loop successfully
- fp32 precision with batch_size=4 is fully stable
- PyTorch detects the GPU correctly; use `torch.cuda.is_available()` (ROCm maps to the CUDA API)

### What doesn't work

**fp16 / mixed precision:** Do not use. fp16 training causes silent crashes and hangs during MIOpen kernel compilation on this architecture.

**Large batch sizes:** batch_size > 4 triggers slow or unstable MIOpen kernel tuning on first run and can silently hang. Stick with batch_size=4 for stability.

**accelerate config:** Use the simplest possible config — no mixed precision, no multi-GPU:

```yaml
# ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
mixed_precision: 'no'
num_processes: 1
```

### MIOpen kernel compilation

On the very first training run, ROCm/MIOpen compiles GPU kernels for each operation. This can take 10–30 minutes before the first step completes. This is normal. Subsequent runs reuse the compiled kernels from cache (`~/.cache/miopen/`).

Do not kill the process during this phase — it will look like a hang but it isn't.

### Performance

With fp32 and batch_size=4 on an 8060S with 137GB unified memory:
- ~105 seconds per training step
- ~4.8 hours per epoch on 11,551 clips
- Full 10-epoch run: ~48 hours