# Plan: GCP GPU Runner for Integration Tests

**Created:** 2026-03-16
**Branch:** feat/gha-gpu-runner
**Status:** Implemented (pending `pulumi up` and PR merge)

## Overview

The python-audio-separator integration tests currently run on a CPU-only self-hosted
GHA runner (`e2-standard-4`, 4 vCPU, 16GB RAM). With the new ensemble tests and
multi-stem verification tests, CI takes 30+ minutes because each model separation runs
on CPU. A GPU runner would reduce this to ~5 minutes.

## Current State

### Existing runner infrastructure
- **Location:** `karaoke-gen/infrastructure/compute/github_runners.py` (Pulumi)
- **Runners:** 3× `e2-standard-4` (general) + 1× `e2-standard-8` (Docker builds)
- **Labels:** `self-hosted`, `Linux`, `X64`, `gcp`, `large-disk`
- **Region:** `us-central1-a`
- **Models:** Pre-cached at `/opt/audio-separator-models` on runner startup
- **Org-level:** Runners are registered to `nomadkaraoke` org, available to all repos
- **NAT:** All runners use Cloud NAT (no external IPs)

### Current integration test workflow
- File: `.github/workflows/run-integration-tests.yaml`
- Runs on: `self-hosted` (picks up any org runner)
- Tests: `poetry run pytest -sv --cov=audio_separator tests/integration`
- Installs: `poetry install -E cpu`
- Problem: All model inference on CPU → very slow for Roformer/Demucs models

## Requirements

- [x] GCE VM with NVIDIA GPU (T4 is cheapest, sufficient for inference)
- [x] CUDA drivers + PyTorch GPU support pre-installed
- [x] Models pre-cached on persistent disk (same as existing runners)
- [x] Labeled `gpu` so workflow can target it specifically
- [x] Cost-effective — only runs when needed (on-demand, not always-on)
- [x] Integration test workflow updated to use `gpu` label
- [x] Install `poetry install -E gpu` (onnxruntime-gpu) instead of `-E cpu`

## Technical Approach

### Option A: Dedicated GPU VM (simplest)

Add a new GPU runner VM to the existing Pulumi infrastructure. Use an `n1-standard-4`
with 1× NVIDIA T4 GPU. Cost: ~$0.35/hr on-demand, ~$0.11/hr spot.

**Pros:** Simple, fits existing patterns, fast startup (VM already running)
**Cons:** Always-on cost if not managed; or slow cold-start if managed

### Option B: Spot GPU VM with startup/shutdown management

Same as A but use spot pricing and the existing runner_manager Cloud Function to
start/stop based on CI demand.

**Pros:** 70% cheaper ($0.11/hr), fits existing management pattern
**Cons:** Spot can be preempted mid-test (rare for short jobs); cold start ~2-3 min

### Option C: Use a cloud GPU service (Modal, Lambda Labs, etc.)

Run the integration tests on a cloud GPU service rather than self-hosted.

**Pros:** No infrastructure to manage, pay-per-second
**Cons:** More complex CI integration, different from existing patterns

### Recommendation: Option B (Spot GPU VM)

The integration test takes <10 minutes on GPU, so spot preemption risk is low.
Cold start is acceptable since it's triggered by PR events. Cost: ~$0.02 per CI run.

## Implementation Steps

### 1. Pulumi infrastructure (in karaoke-gen repo)

1. [x] Add `GITHUB_GPU_RUNNER` machine type to `config.py`: `n1-standard-4` + 1× T4
2. [x] Add `GPU_RUNNER_LABELS` to `config.py`: `"self-hosted,linux,x64,gcp,gpu"`
3. [x] Create GPU runner VM in `github_runners.py`:
   - `n1-standard-4` (4 vCPU, 15GB RAM)
   - 1× NVIDIA T4 GPU (`nvidia-tesla-t4`)
   - `guest_accelerators` config
   - `on_host_maintenance: "TERMINATE"` (required for GPU VMs)
   - Same NAT/networking as existing runners
4. [x] Create GPU startup script (`github_runner_gpu.sh`):
   - Install NVIDIA drivers via CUDA repo (cuda-drivers + cuda-toolkit-12-4)
   - Install CUDA toolkit
   - Verify GPU: `nvidia-smi`
   - Pre-download models to `/opt/audio-separator-models`
   - Register as GHA runner with `gpu` label
5. [x] Add spot scheduling for cost optimization
6. [ ] Run `pulumi up` to create the VM

### 2. Workflow update (in python-audio-separator repo)

7. [x] Update `run-integration-tests.yaml`:
   - Change `runs-on: self-hosted` to `runs-on: [self-hosted, gpu]`
   - Change `poetry install -E cpu` to `poetry install -E gpu`
   - Add `nvidia-smi` verification step
   - Add 30-minute timeout
8. [ ] Add fallback: if no GPU runner available, fall back to CPU with longer timeout
   - Deferred: not needed initially, the runner_manager auto-starts the GPU VM on demand

### 3. Startup script details

The GPU startup script needs to:
```bash
# Install NVIDIA drivers (for Debian 12)
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) nvidia-driver-535

# Verify GPU
nvidia-smi

# Install CUDA (for PyTorch)
# PyTorch bundles its own CUDA, so we mainly need the driver

# Pre-download models
pip install audio-separator[gpu]
python -c "
from audio_separator.separator import Separator
sep = Separator(model_file_dir='/opt/audio-separator-models')
# Download all models used in integration tests
models = [
    'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    'MGM_MAIN_v4.pth',
    'UVR-MDX-NET-Inst_HQ_4.onnx',
    'kuielab_b_vocals.onnx',
    '2_HP-UVR.pth',
    'htdemucs_6s.yaml',
    'htdemucs_ft.yaml',
    # Ensemble preset models
    'bs_roformer_vocals_resurrection_unwa.ckpt',
    'melband_roformer_big_beta6x.ckpt',
    'bs_roformer_vocals_revive_v2_unwa.ckpt',
    'mel_band_roformer_kim_ft2_bleedless_unwa.ckpt',
    'bs_roformer_vocals_revive_v3e_unwa.ckpt',
    'mel_band_roformer_vocals_becruily.ckpt',
    'mel_band_roformer_vocals_fv4_gabox.ckpt',
    'mel_band_roformer_instrumental_fv7z_gabox.ckpt',
    'bs_roformer_instrumental_resurrection_unwa.ckpt',
    'melband_roformer_inst_v1e_plus.ckpt',
    'mel_band_roformer_instrumental_becruily.ckpt',
    'mel_band_roformer_instrumental_instv8_gabox.ckpt',
    'UVR-MDX-NET-Inst_HQ_5.onnx',
    'mel_band_roformer_karaoke_gabox_v2.ckpt',
    'mel_band_roformer_karaoke_becruily.ckpt',
    # Multi-stem test models
    '17_HP-Wind_Inst-UVR.pth',
    'MDX23C-DrumSep-aufr33-jarredou.ckpt',
    'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
]
for m in models:
    sep.download_model_and_data(m)
"
```

## Cost Estimate

| Config | Hourly | Per CI run (~10 min) | Monthly (est. 100 runs) |
|--------|--------|---------------------|-------------------------|
| n1-standard-4 + T4 (on-demand) | $0.61 | $0.10 | $10 |
| n1-standard-4 + T4 (spot) | $0.19 | $0.03 | $3 |
| Current CPU (e2-standard-4) | $0.13 | $0.07 | $7 |

Spot GPU is actually cheaper per-run than current CPU because GPU tests finish 5× faster.

## Files to Create/Modify

| File | Repo | Action |
|------|------|--------|
| `infrastructure/config.py` | karaoke-gen | Add GPU machine type + labels |
| `infrastructure/compute/github_runners.py` | karaoke-gen | Add GPU runner VM |
| `infrastructure/compute/startup_scripts/github_runner_gpu.sh` | karaoke-gen | GPU-specific startup |
| `.github/workflows/run-integration-tests.yaml` | python-audio-separator | Target GPU runner |

## Open Questions

- [x] Should the GPU runner be spot or on-demand? → **Spot** ($0.19/hr, ~$3/mo)
- [x] Should we keep the CPU fallback for when GPU runner is unavailable? → **Deferred** (runner_manager auto-starts VM)
- [x] Should the runner startup script install NVIDIA drivers from scratch each boot,
      or use a pre-built GCP Deep Learning VM image? → **From scratch** (idempotent, matches existing pattern)
- [x] Zone availability: T4 GPUs may not be available in us-central1-a → **Available** in all us-central1 zones (a, b, c, f)

## Rollback Plan

The GPU runner is additive infrastructure. If it fails:
1. Change workflow back to `runs-on: self-hosted` (CPU)
2. Destroy the GPU VM via `pulumi destroy` targeting just that resource
