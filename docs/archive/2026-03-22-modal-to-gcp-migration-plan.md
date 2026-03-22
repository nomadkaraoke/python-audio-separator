# Plan: Modal → GCP Audio Separation Migration

**Created:** 2026-03-22
**Branch:** feat/sess-20260321-2314-modal-gcp-migration
**Worktrees:** `karaoke-gen-modal-gcp-migration` (infra + backend), `python-audio-separator-modal-gcp-migration` (server)
**Status:** Draft → Ready for implementation

## Overview

Migrate audio stem separation from Modal to a Cloud Run Service with L4 GPU on GCP. This eliminates the only third-party compute dependency, fixes intermittent Modal API failures ("no files were downloaded"), upgrades to latest ensemble models for better quality, and decouples separation from the lyrics review critical path so users can start reviewing lyrics faster.

### Architecture Decision: Cloud Run GPU Service

| Factor | Cloud Run GPU | GCE VM + auto-stop |
|--------|--------------|-------------------|
| Idle cost | $0 (scales to zero) | $0 (when stopped) |
| Cold start | ~30-60s (model load from GCS) | ~60-120s (VM boot + model load) |
| Ops overhead | None (serverless) | Moderate (start/stop scripts, health monitoring) |
| GPU available | L4 (24GB VRAM) in us-central1 | T4/L4/A100 |
| Scaling | Automatic | Manual orchestration |
| Cost/job (~12 min GPU) | ~$0.13 | ~$0.07-0.10 (T4) |
| Deployment | Docker image push | Packer image + GCS wheel + SSH restart |

Cloud Run GPU wins on simplicity. L4 is faster than T4, cold start is acceptable, and per-job cost well under $1.

### Model Upgrade: Ensemble Presets as Default

**Current models (single-model):**
| Stage | Model | SDR | Notes |
|-------|-------|-----|-------|
| 1 (instrumental) | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 12.97 | Older BS-Roformer |
| 1 (other stems) | `htdemucs_6s.yaml` | — | Demucs 6-stem — **dropping** |
| 2 (karaoke/BV) | `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | 10.20 | Single karaoke model |

**New defaults (ensemble presets):**
| Stage | Preset | Models | SDR | Notes |
|-------|--------|--------|-----|-------|
| 1 | `instrumental_clean` | Fv7z + Resurrection | ~17.5 | +35% quality, bleedless |
| 2 | `karaoke` | 3 karaoke models (aufr33+gabox_v2+becruily) | ~10.6 | +4% quality, 3-model ensemble |

**Key design: preset-name references, not model filenames.** karaoke-gen references preset names (`instrumental_clean`, `karaoke`). The audio-separator package resolves preset → models + ensemble algorithm. When better models come out, update presets in audio-separator and release a new version — no karaoke-gen changes needed.

### Pipeline Decoupling: Separation Off Critical Path

**Current flow (both gate review):**
```
Job created
├── Audio worker (separation) ──→ audio_complete=True ─┐
│                                                       ├→ GENERATING_SCREENS → AWAITING_REVIEW
└── Lyrics worker (transcription) → lyrics_complete=True┘
```

**New flow (lyrics gates review, separation runs in background):**
```
Job created
├── Audio worker (separation) ──→ audio_complete=True (background, not gating)
│
└── Lyrics worker (transcription) → lyrics_complete=True → GENERATING_SCREENS → AWAITING_REVIEW
                                                                                     │
                                                                              User reviews lyrics
                                                                                     │
                                                                              Instrumental review
                                                                              (waits for audio_complete
                                                                               if not ready yet)
```

**Why this works:**
- Lyrics review (`/app/jobs#/{jobId}/review`) only needs transcription output — no stems needed
- Instrumental review (`/app/jobs#/{jobId}/instrumental`) needs stems — but user typically spends 5+ min on lyrics review, buying time for separation to finish
- In the rare case separation isn't done when user reaches instrumental review, show a "Separation in progress..." waiting state
- Screens worker only truly needs lyrics to generate title/end screens

### Estimated Timeline

| Scenario | Stage 1 | Stage 2 | Cold start | Total |
|----------|---------|---------|------------|-------|
| Current (Modal, single models) | 3-5 min | 2-3 min | 0 | 7-11 min |
| New ensemble (Cloud Run L4) | ~4-6 min | ~3-5 min | ~30-60s | ~8-12 min |
| **User-perceived wait** (new) | — | — | — | **0 min** (decoupled) |

Separation takes slightly longer with ensembles, but users never wait for it — they're reviewing lyrics while it runs.

## Requirements

- [ ] Audio separation runs on GCP Cloud Run with L4 GPU
- [ ] Same HTTP API contract as Modal deployment (endpoints, request/response format)
- [ ] `audio-separator-remote` CLI and `AudioSeparatorAPIClient` work unchanged
- [ ] Default models use ensemble presets (`instrumental_clean` + `karaoke`)
- [ ] karaoke-gen references preset names, not model filenames
- [ ] Demucs 6-stem separation dropped from pipeline
- [ ] Scale-to-zero when not processing (no idle GPU cost)
- [ ] Cold start < 60 seconds
- [ ] Per-job cost < $1
- [ ] Models stored in GCS, loaded on container startup
- [ ] Publicly accessible endpoint with auth token (reuse `admin-tokens` secret)
- [ ] Infrastructure managed via Pulumi in karaoke-gen
- [ ] Separation decoupled from lyrics review critical path
- [ ] Instrumental review page handles "separation still in progress" gracefully
- [ ] Docker image CI lives in python-audio-separator repo, pushes to Artifact Registry

## Implementation Steps

### Phase 1: Cloud Run GPU Server (python-audio-separator repo)

#### Step 1.1 — Create Cloud Run-compatible FastAPI server
- [ ] Create `audio_separator/remote/deploy_cloudrun.py` adapted from `deploy_modal.py`
- [ ] Replace Modal-specific code:
  - `modal.Dict` → in-memory `dict` (single instance handles one job at a time)
  - `modal.Volume` → local `/tmp` storage + GCS for model cache
  - `modal.Function.spawn()` → synchronous processing (no background tasks needed)
  - `modal.Image` → Dockerfile
  - `modal.App` → standard FastAPI + uvicorn
- [ ] Keep all existing API endpoints identical:
  - `POST /separate` — submit separation job
  - `GET /status/{task_id}` — return job status
  - `GET /download/{task_id}/{file_hash}` — download result file
  - `GET /models-json`, `GET /models` — list models
  - `GET /health` — health check (with model readiness indicator)
  - `GET /` — root info
- [ ] Add model download on startup from GCS bucket (`gs://nomadkaraoke-audio-separator-models/`)
- [ ] Add ensemble preset support: accept `preset` parameter in `/separate` that resolves to model list + algorithm
- [ ] Add startup probe endpoint for Cloud Run GPU readiness

**Design:** Make `/separate` effectively synchronous — process inline, store results in-memory dict + local filesystem. Cloud Run instance stays alive for scale-down timeout (600s), so Stage 2 hits the same warm instance. Async polling API contract preserved for client compatibility.

#### Step 1.2 — Create Dockerfile
- [ ] Create `Dockerfile.cloudrun` in repo root
- [ ] Base: `nvidia/cuda:12.6.3-runtime-ubuntu22.04` (matches Cloud Run L4 driver support)
- [ ] Install: Python 3.13, FFmpeg, libsndfile, sox, system audio libs
- [ ] Install: `audio-separator[gpu]` from current repo
- [ ] Entrypoint: `python -m audio_separator.remote.deploy_cloudrun`
- [ ] Expose port 8080
- [ ] Set env: `MODEL_DIR=/models`, `STORAGE_DIR=/tmp/storage`

#### Step 1.3 — Upload models to GCS
- [ ] Create GCS bucket `nomadkaraoke-audio-separator-models` (us-central1, standard storage)
- [ ] Upload all models needed by default ensemble presets:
  - `mel_band_roformer_instrumental_fv7z_gabox.ckpt` (instrumental_clean preset)
  - `bs_roformer_instrumental_resurrection_unwa.ckpt` (instrumental_clean preset)
  - `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` (karaoke preset)
  - `mel_band_roformer_karaoke_gabox_v2.ckpt` (karaoke preset)
  - `mel_band_roformer_karaoke_becruily.ckpt` (karaoke preset)
- [ ] Total: ~1-1.5 GB of models

#### Step 1.4 — Local testing
- [ ] Build Docker image locally
- [ ] Test with `docker run --gpus all` (if local GPU available) or CPU mode
- [ ] Verify API compatibility: submit job with `preset=instrumental_clean`, poll status, download files
- [ ] Verify output filename format matches expected pattern: `filename_(StemType)_modelname.ext`
- [ ] Verify ensemble output: ensembled stems have correct naming
- [ ] Compare output quality with Modal (A/B test on reference songs)

#### Step 1.5 — CI/CD for Docker image
- [ ] Create `.github/workflows/deploy-to-cloudrun.yml` in python-audio-separator repo
- [ ] Triggers: PyPI release, changes to Dockerfile.cloudrun, manual dispatch
- [ ] Steps: build Docker image → push to Artifact Registry (`us-central1-docker.pkg.dev/nomadkaraoke/audio-separator`)
- [ ] Use Workload Identity Federation for GCP auth

### Phase 2: GCP Infrastructure (karaoke-gen repo)

#### Step 2.1 — Artifact Registry
- [ ] Add Artifact Registry Docker repo to Pulumi
- [ ] Repository: `audio-separator` in `us-central1`

#### Step 2.2 — GCS Model Bucket
- [ ] Create `nomadkaraoke-audio-separator-models` bucket via Pulumi
- [ ] Standard storage class, us-central1
- [ ] Grant read access to Cloud Run service account

#### Step 2.3 — Cloud Run GPU Service
- [ ] Create `infrastructure/modules/audio_separator_service.py`
- [ ] Cloud Run Service configuration:
  - Image: from Artifact Registry
  - GPU: 1x NVIDIA L4
  - CPU: 4 vCPU (minimum required for L4)
  - Memory: 16 GiB
  - Min instances: 0 (scale to zero)
  - Max instances: 2 (handle concurrent jobs)
  - Request timeout: 1800s (30 min)
  - Scale-down delay: 600s (keep warm between Stage 1 → Stage 2)
  - Startup probe: HTTP GET /health, 120s initial delay, 10s period
  - Env vars:
    - `MODEL_BUCKET=nomadkaraoke-audio-separator-models`
    - `MODEL_DIR=/models`
    - `ADMIN_TOKEN` (from Secret Manager, reuse existing `admin-tokens`)
  - Region: us-central1
  - Ingress: all traffic (public endpoint with auth)

#### Step 2.4 — Service Account & IAM
- [ ] Create `audio-separator` service account
- [ ] Grant: `storage.objectViewer` on model bucket
- [ ] Grant: `secretmanager.secretAccessor` for admin-tokens
- [ ] Grant: `logging.logWriter`, `monitoring.metricWriter`

#### Step 2.5 — Wire into Pulumi
- [ ] Add to `infrastructure/__main__.py`
- [ ] Add config constants to `infrastructure/config.py`

### Phase 3: Pipeline Decoupling + Model Upgrade (karaoke-gen repo)

#### Step 3.1 — Decouple separation from lyrics review path
- [ ] In `backend/services/job_manager.py`:
  - Change `check_parallel_processing_complete()` to only check `lyrics_complete` (not `audio_complete`)
  - `mark_lyrics_complete()` triggers screens worker on its own (no need to wait for audio)
  - `mark_audio_complete()` no longer triggers screens — just sets the flag
- [ ] In `backend/workers/screens_worker.py`:
  - Remove validation that `audio_complete` must be True
  - Screens only needs lyrics data to generate title/end screens
  - Skip instrumental analysis step if audio isn't complete yet (or make it a no-op)
- [ ] Verify: lyrics review page works without stems present

#### Step 3.2 — Add "waiting for separation" state to instrumental review
- [ ] In frontend instrumental review page (`/app/jobs#/{jobId}/instrumental`):
  - Check `state_data.audio_complete` on page load
  - If false, show "Audio separation in progress..." with a spinner/progress indicator
  - Poll job status every 5-10 seconds until `audio_complete=True`
  - Once complete, load and display instrumental options as normal
- [ ] Backend: ensure instrumental review API endpoint returns separation status

#### Step 3.3 — Switch to preset-based model configuration
- [ ] In `backend/workers/audio_worker.py`:
  - Replace `DEFAULT_CLEAN_MODEL` with `DEFAULT_INSTRUMENTAL_PRESET = "instrumental_clean"`
  - Replace `DEFAULT_BACKING_MODELS` with `DEFAULT_KARAOKE_PRESET = "karaoke"`
  - Remove `DEFAULT_OTHER_MODELS` (Demucs dropped)
  - Pass `preset=` parameter to API client instead of `models=`
- [ ] In `karaoke_gen/audio_processor.py`:
  - Update `_process_audio_separation_remote()` to pass presets
  - Stage 1: `api_client.separate_audio_and_wait(audio_file, preset="instrumental_clean", ...)`
  - Stage 2: `api_client.separate_audio_and_wait(vocals_file, preset="karaoke", ...)`
  - Remove `other_stems_models` parameter (or default to empty)
  - Update result organization for ensemble outputs (stem names may include ensemble info)
- [ ] In `audio_separator/remote/api_client.py` (python-audio-separator repo):
  - Add `preset` parameter to `separate_audio()` and `separate_audio_and_wait()`
  - Client passes `preset` field in multipart form data to API
  - API server resolves preset → models + algorithm

#### Step 3.4 — Update tests
- [ ] Update `tests/unit/test_audio_remote.py`:
  - Test preset-based separation calls
  - Remove Demucs 6-stem references
  - Test new default model/preset names
- [ ] Add test for pipeline decoupling:
  - Verify `mark_lyrics_complete()` triggers screens without `audio_complete`
  - Verify `mark_audio_complete()` sets flag but doesn't trigger screens
- [ ] Add frontend test for instrumental review waiting state

### Phase 4: Cutover & Cleanup

#### Step 4.1 — Deploy and test
- [ ] Deploy Cloud Run GPU service via `pulumi up`
- [ ] Run separation on 3-5 test songs with ensemble presets
- [ ] Compare output quality to Modal (listen test)
- [ ] Verify timing: ensemble separation completes within ~8-12 min
- [ ] Test cold start scenario (wait for scale-down, then submit)
- [ ] Test back-to-back jobs (Stage 1 → Stage 2 hits warm instance)
- [ ] Test pipeline decoupling: verify lyrics review available before separation completes
- [ ] Test instrumental review waiting state

#### Step 4.2 — Update Cloud Run audio worker config
- [ ] Change `AUDIO_SEPARATOR_API_URL` in `infrastructure/modules/cloud_run.py` from Modal URL to Cloud Run URL
- [ ] Deploy via `pulumi up`
- [ ] Run 5-10 production jobs, monitor for errors

#### Step 4.3 — Monitor (1 week)
- [ ] Watch Cloud Run logs for errors
- [ ] Monitor separation timing in job state_data
- [ ] Check Cloud Run billing (verify per-job cost < $1)
- [ ] Verify scale-to-zero works (no idle GPU charges)
- [ ] Watch for users hitting the "waiting for separation" state — measure frequency

#### Step 4.4 — Decommission Modal
- [ ] Remove Modal deployment workflow from python-audio-separator repo
- [ ] Delete Modal app
- [ ] Close Modal account
- [ ] Remove `modal` from python-audio-separator dependencies
- [ ] Update `AUDIO_SEPARATOR_API_URL` env var in local `.envrc` files

## Files to Create/Modify

### python-audio-separator repo (`python-audio-separator-modal-gcp-migration` worktree)
| File | Action | Description |
|------|--------|-------------|
| `audio_separator/remote/deploy_cloudrun.py` | Create | Cloud Run-compatible FastAPI server (adapted from deploy_modal.py) |
| `audio_separator/remote/api_client.py` | Modify | Add `preset` parameter to separate methods |
| `Dockerfile.cloudrun` | Create | Docker image for Cloud Run GPU deployment |
| `.github/workflows/deploy-to-cloudrun.yml` | Create | CI/CD: build image → push to Artifact Registry |

### karaoke-gen repo (`karaoke-gen-modal-gcp-migration` worktree)
| File | Action | Description |
|------|--------|-------------|
| `infrastructure/modules/audio_separator_service.py` | Create | Pulumi: Cloud Run GPU service + model bucket + IAM |
| `infrastructure/__main__.py` | Modify | Wire up audio separator service |
| `infrastructure/config.py` | Modify | Add audio separator constants |
| `infrastructure/modules/cloud_run.py` | Modify | Update `AUDIO_SEPARATOR_API_URL` to Cloud Run URL |
| `backend/services/job_manager.py` | Modify | Decouple: lyrics_complete alone triggers screens |
| `backend/workers/screens_worker.py` | Modify | Remove audio_complete prerequisite |
| `backend/workers/audio_worker.py` | Modify | Switch to preset-based config, drop Demucs |
| `karaoke_gen/audio_processor.py` | Modify | Pass presets instead of model filenames |
| `frontend/` (instrumental review) | Modify | Add "waiting for separation" state |
| `tests/unit/test_audio_remote.py` | Modify | Update for presets, remove Demucs tests |
| `.github/workflows/deploy-audio-separator.yml` | Create | CI: deploy Cloud Run revision on image push |

## Testing Strategy

- **Unit tests:** Preset resolution, pipeline decoupling (lyrics triggers screens alone), model name updates
- **Integration test:** Deploy Cloud Run service, run full separation with ensemble presets, verify output files
- **A/B comparison:** Same songs through Modal (single model) and Cloud Run (ensemble) — quality should be better
- **Pipeline test:** Submit job, verify lyrics review available before separation completes
- **Frontend test:** Playwright E2E for instrumental review waiting state
- **Cold start test:** Wait for scale-down, submit job, measure total time
- **Production E2E:** After cutover, run 10 production jobs through full pipeline

## Cost Estimate

| Scenario | Monthly cost |
|----------|-------------|
| 10 jobs/day × 12 min GPU = 2 hrs/day | ~$40/mo |
| 30 jobs/day × 12 min GPU = 6 hrs/day | ~$120/mo |
| Per-job cost (12 min L4 @ $0.67/hr) | ~$0.13 |

Well under $1/job budget. No idle cost due to scale-to-zero.

## Resolved Questions

- [x] Cloud Run vs GCE VM → **Cloud Run GPU Service** (simplest, scale-to-zero)
- [x] Which GPU → **L4** (only option on Cloud Run, 24GB VRAM)
- [x] Model upgrade → **Ensemble presets as default** (quality > speed)
- [x] Demucs 6-stem → **Drop it**
- [x] Auth → **Reuse existing `admin-tokens` secret**
- [x] Docker CI → **python-audio-separator repo builds + pushes image**
- [x] Ensemble presets UI → **Backend-only; presets defined in audio-separator package**
- [x] Speed vs quality → **Quality wins; decouple separation from critical path so user never waits**

## Rollback Plan

1. **Quick rollback:** Change `AUDIO_SEPARATOR_API_URL` back to Modal URL in Pulumi config, `pulumi up`. Takes ~2 minutes.
2. **Pipeline rollback:** Revert job_manager changes to re-gate screens on `audio_complete`. One commit.
3. **Keep Modal running** during the monitoring period (Phase 4.3). Don't decommission until confident.
4. **Model rollback:** Preset config can be changed back to direct model filenames in one commit.
