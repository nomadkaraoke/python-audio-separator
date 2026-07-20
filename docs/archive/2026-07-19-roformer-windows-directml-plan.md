# Plan: RoFormer on Windows / DirectML (issue #292)

**Date:** 2026-07-19
**Branch:** `feat/sess-20260719-2104-roformer-windows-directml`
**Issue:** https://github.com/nomadkaraoke/python-audio-separator/issues/292

## Goal

Make RoFormer (BSRoformer + MelBandRoformer) models work with `--use_directml` on
Windows AMD/Intel GPUs, with **zero behavioral change** for CUDA, MPS, and CPU users
on any platform. Add Windows + DirectML coverage to CI so Windows becomes a
genuinely supported platform ongoing.

Demucs on DirectML is **out of scope**: it fails on `aten::_thnn_fused_lstm_cell`,
a missing operator in torch-directml itself. Nothing we can patch; document as
unsupported in the README compatibility table.

## Root-cause analysis (from issue report + code reading)

The reporter (torch-directml 0.2.5, audio-separator 0.44.2) hit two distinct failures:

### Failure 1: `'>=' not supported between instances of 'torch.device' and 'int'`

`RoformerLoader._load_with_new_implementation` calls
`torch.load(model_path, map_location=device)` (`roformer_loader.py:98`) where
`device` is `str(self.torch_device)` = `"privateuseone:0"` (passed from
`mdxc_separator.py:92`). torch-directml registers a deserialization hook for the
`privateuseone` backend that expects an integer device id; handing it a
`torch.device` triggers the TypeError inside `torch_directml.device()`'s
`device_id >= device_count()` comparison.

Consequence: the **new** implementation always fails on DML and silently falls
back to the legacy path (which loads with `map_location='cpu'` — that's why
legacy gets further). So DML users currently never get the new implementation.

Note the non-Roformer MDXC path already knows about this class of problem —
`mdxc_separator.py:~108` loads TFC_TDF state dicts with `map_location="cpu"` with
the comment "loading the state onto a hardware accelerated device causes issues".

### Failure 2: `Invalid or unsupported data type ComplexFloat`

torch-directml has **no complex tensor support**. The Roformer forward pass uses
complex ops on the model device (`uvr_lib_v5/roformer/bs_roformer.py`, same
pattern in `mel_band_roformer.py`):

- `torch.stft(..., return_complex=True)` on input (bs_roformer.py:453)
- `torch.view_as_complex` + complex mask multiplication (bs_roformer.py:503–506)
- `torch.istft` (bs_roformer.py:512)

The codebase already contains both precedents for the fix:

- **MDX arch** (`uvr_lib_v5/stft.py:22`): "is_non_standard_device" (device type
  not cpu/cuda) → hop tensor to CPU for STFT/ISTFT, move result back. This is
  exactly why MDX **works** on DML today.
- **Roformer MPS path** (bs_roformer.py:512): `x_is_mps` → istft runs on CPU.

### Likely failure 3 (undiscovered — blocked behind failure 2)

`Attend.flash_attn` (`uvr_lib_v5/roformer/attend.py`) calls
`F.scaled_dot_product_attention` inside a `torch.backends.cuda.sdp_kernel`
context; `flash_attn` defaults to `True` in the loader config. SDPA support on
torch-directml is doubtful. If it fails, `Attend.forward` already has a pure
einsum/softmax math path we can route DML tensors to. We won't know until the
complex-op fix lets inference reach the transformer — the DML CI job (Phase 4)
and the issue reporter will tell us.

## Fix design — everything gated on `device.type == "privateuseone"`

Guiding principle: every change is behind an explicit DirectML check so
CUDA/MPS/CPU code paths are **byte-identical** before and after. The MPS istft
hop keeps its own `x_is_mps` flag — we do NOT merge MPS into a generic
"non-standard device" branch, because MPS currently runs the forward STFT
on-device and that must not change.

Add one tiny helper (e.g. in `uvr_lib_v5/roformer/` or `common_separator`):

```python
def is_dml_device(device) -> bool:
    return torch.device(device).type == "privateuseone"
```

(`privateuseone` is torch's generic out-of-tree backend slot, so in theory a
non-DML backend could claim it — but this branch is only reachable when the
Separator itself set the device via `torch_directml.device()` under
`use_directml=True`, so the check cannot misfire for cuda/mps/cpu users.)

### Phase 2a: loader fix (kills failure 1)

In `RoformerLoader._load_with_new_implementation`:

```python
map_loc = 'cpu' if is_dml_device(device) else device
state_dict = torch.load(model_path, map_location=map_loc)
...
model.to(device)   # unchanged; torch.device("privateuseone:0") is valid once torch_directml is imported
```

Gated on DML only (conservative), though CPU-then-`.to()` is semantically
identical everywhere. Same treatment in `_load_with_legacy_implementation`
already exists (it uses `'cpu'` unconditionally) — no change there. Keep the
legacy fallback mechanism fully intact.

Also verify `model.to("privateuseone:0")` works when given the *string* — the
legacy path already does `.to(device)` with the same string and reached
inference in the reporter's log, so this is confirmed working.

### Phase 2b: complex-op CPU hops in `bs_roformer.py` + `mel_band_roformer.py`

In `forward()`, compute `x_is_dml = original_device.type == "privateuseone"` next
to the existing `x_is_mps`:

1. **Forward STFT**: if `x_is_dml`, run `torch.stft` on CPU
   (`raw_audio.cpu()`, CPU window), `view_as_real`, then move the now-real
   tensor to the DML device. The band-split + transformer stack (the heavy
   compute, i.e. the part worth accelerating) stays on DML.
2. **Mask modulation + iSTFT**: if `x_is_dml`, move `stft_repr` and `mask` to
   CPU before `view_as_complex`, do the complex multiply + `torch.istft` on
   CPU, move `recon_audio` back to the device (the demix loop pulls outputs to
   CPU immediately anyway, so this costs almost nothing).
3. Existing `x_is_mps` istft condition becomes `x_is_mps or x_is_dml` **only**
   at the istft call where MPS already hops — the extra DML-only hops (stft,
   complex multiply) do not touch the MPS branch.

Apply identically to both model files; their forward passes are near-clones.

### Phase 2c: attention fallback (contingent on failure 3 materializing)

If SDPA fails on DML: in `Attend.forward`, route
`q.device.type == "privateuseone"` to the existing non-flash einsum path
(einsum/softmax are basic ops torch-directml supports). Do NOT change flash
behavior for cuda/cpu/mps.

## Testing & CI plan

### Current state

- `run-unit-tests.yaml`: ubuntu + macos + **windows-latest** (CPU, `-E cpu`),
  Python 3.10–3.13 — Windows unit coverage already exists.
- `run-integration-tests.yaml`: 3 jobs on self-hosted **Linux** GCP T4 runners
  (`[self-hosted, gpu]`), pre-cached models at `/opt/audio-separator-models`.
  Branch-protection ruleset 529535 requires `ensemble-presets`, `core-models`,
  `stems-and-quality`, `unit-tests` — **any new required job name must be added
  to the ruleset** (docs/CI-GPU-RUNNERS.md gotcha).

### New CI tier 1 — Windows CPU integration smoke (free, every PR)

New job `windows-cpu-integration` on `windows-latest` in
`run-integration-tests.yaml` (path-filtered like the others):

- `poetry install -E cpu`; ffmpeg via choco or `FedericoCarboni/setup-ffmpeg`.
- Run a **short-fixture** e2e subset: one small RoFormer model (pick the
  smallest mel-band checkpoint used by existing tests) + one MDX model on a
  few-second test clip. CPU RoFormer on 4 cores is slow — the fixture length is
  the lever; target < 10 min wall clock.
- Model download cached with `actions/cache` keyed on model filenames
  (no pre-cached model dir on hosted runners).
- Purpose: catches Windows-specific breakage (paths, soundfile/ffmpeg,
  libsamplerate) on every PR, forever. This is the "Windows is a supported
  platform" baseline, independent of DirectML.
- Land this FIRST (before code changes) so we have a green Windows baseline.

### New CI tier 2 — Windows DirectML on the existing ephemeral runner fleet

**Decision (Andrew, 2026-07-19):** follow the existing ephemeral self-hosted
runner pattern and extend it with Windows support, rather than GitHub-hosted
GPU larger runners. The new job becomes part of the existing mandatory
integration checks (ruleset 529535) once stable.

#### How the current fleet works (karaoke-gen infra)

- `workflow_job.queued` org webhook → `github-runner-manager` Cloud Function
  (`karaoke-gen/infrastructure/functions/runner_manager/{main,ephemeral}.py`)
  mints a JIT config and creates a **single-use** GCE VM from a pre-baked
  image family. VM runs one job (`./run.sh --jitconfig`), self-shuts-down via
  an EXIT trap; a 15-min orphan-cleanup scheduler pass deletes stopped VMs and
  de-registers zombies.
- Families are defined in `ephemeral.py` `FAMILIES` (`general`/`build`/`gpu`),
  resolved from job labels (`resolve_family`, precedence gpu → build →
  general). Advertised labels come from `runner_labels_for()` — which
  currently **hardcodes `linux`**.
- Images are baked by `karaoke-gen/.github/workflows/build-runner-images.yml`
  (monthly cron + manual dispatch): temp VM from `debian-12` runs
  `infrastructure/scripts/runner-image-provision.sh`, then the boot disk is
  snapshotted into the `gha-runner-<variant>` image family. The GPU image
  pre-bakes NVIDIA driver + ~14GB of audio-separator models on a 200GB disk.

#### Windows extension (changes in karaoke-gen repo)

1. **New family** `gpu-windows` in `ephemeral.py` `FAMILIES`:
   - `machine_type="n1-standard-4"`, T4 accelerator, `disk_size_gb=200`,
     `image_family="gha-runner-gpu-windows"`.
   - Add an `os_label` field to `FamilySpec` (default `"linux"`, `"windows"`
     for the new family) so `runner_labels_for()` stops hardcoding `linux`.
     Advertised labels: `self-hosted, windows, x64, gcp, gpu`.
   - `resolve_family`: check `windows` first — `windows`+`gpu` → `gpu-windows`
     (plain `windows` without gpu can 404/fall through for now; we have no
     non-GPU Windows family).
   - Windows VMs take startup scripts via the **`windows-startup-script-ps1`**
     metadata key, not `startup-script` — `_build_instance` needs a
     family-conditional PowerShell equivalent of `STARTUP_SCRIPT`: fetch
     `jit-config` from instance metadata, `cd C:\actions-runner`,
     `.\run.cmd --jitconfig $jit`, then `shutdown /s` in a finally block.
   - Cost: T4 + n1-standard-4 (~$0.54/hr) + Windows Server license
     (~$0.18/hr) ≈ **$0.72/hr → ~$0.18 per 15-min run**. Scale-to-zero
     unchanged; orphan cleanup works as-is (it's OS-agnostic: reconciles VM
     list vs org runner registrations).
   - Update `test_ephemeral.py` for the new family/labels/metadata logic.

2. **New bake variant** in `build-runner-images.yml` +
   `runner-image-provision.ps1`:
   - Base image: `windows-cloud/windows-server-2022-dc` (not debian).
   - **Driver gotcha:** install the **NVIDIA GRID/virtual-workstation driver**
     (GCP hosts installers in `gs://nvidia-drivers-us-public`), NOT the
     datacenter/CUDA driver. T4 under the datacenter driver runs in TCC
     compute mode with no DirectX/WDDM support — DirectML would see no GPU at
     all. GRID driver → WDDM → DX12 → DirectML works.
   - Provision: Python **3.12** (torch-directml has no 3.13 wheels — the
     issue reporter's working env is 3.12.8; verify at implementation time),
     git, ffmpeg, Poetry, actions-runner **win-x64** package at
     `C:\actions-runner`, pre-baked model cache (same model set as the Linux
     GPU image) at `C:\audio-separator-models`.
   - Readiness signaling: the Linux flow polls for `/opt/runner-image-ready`
     via SSH; for Windows, write a marker line to the serial console and poll
     `gcloud compute instances get-serial-port-output` (no SSH/WinRM
     dependency — matches the repo's existing "serial console is the only
     reliable diagnostic channel" convention).

#### Workflow changes (python-audio-separator repo)

- **Label-routing fix (must land before/with the Windows runner):** the three
  existing jobs use `runs-on: [self-hosted, gpu]`. GitHub schedules a job onto
  any runner whose labels are a *superset* — a Windows GPU runner
  (`self-hosted, windows, x64, gcp, gpu`) would match and could steal Linux
  jobs. Change existing jobs to `runs-on: [self-hosted, linux, gpu]` (Linux
  runners already advertise `linux`, so this is a no-op for them today).
- **New job `windows-directml`** with `runs-on: [self-hosted, windows, gpu]`,
  `AUDIO_SEPARATOR_MODEL_DIR: C:\audio-separator-models`, Python 3.12,
  `poetry install -E dml`. Initially `continue-on-error: true` / not in the
  gate job; promoted to required once stable.
- Job content:
  - Assert `--env_info` reports DirectML available in Torch +
    `DmlExecutionProvider` in ONNX Runtime.
  - Matrix of supported-on-DML architectures with `use_directml=True`:
    MDX (`UVR-MDX-NET-Inst_HQ_3.onnx`) and VR (`5_HP-Karaoke-UVR.pth`) as
    regression guards for what already works, BSRoformer
    (`model_bs_roformer_ep_317_sdr_12.9755.ckpt` — the exact model from the
    issue) + one MelBandRoformer as the new coverage.
  - **Assert the NEW implementation loaded** (no "Fell back to legacy" /
    check `ModelLoadingResult` implementation) — otherwise the map_location
    fix can silently regress while everything still "passes" via fallback.
  - **Quality assertion, not just no-crash**: separate the same fixture on
    CPU and on DML in the same job; assert (a) correlation / SDR-style
    similarity vs the CPU output, (b) metrics are finite (no NaN/Inf), and
    (c) an explicit RMS/peak energy floor on the DML output — silent or
    near-silent output must fail even if a similarity metric degenerates.
    Reuse/extend helpers from `test_roformer_audio_quality.py`.
- Note: T4/NVIDIA validates the DirectML *code path* (DX12), which is the
  right CI proxy even though the reporter has an AMD iGPU — final AMD
  validation comes from the reporter (below).

### Unit tests (all platforms, free)

- `RoformerLoader`: `map_location` is `'cpu'` when device is
  `privateuseone:*`, unchanged (`device` passthrough) for `cpu`/`cuda:0`/`mps`
  (mock `torch.load`, assert call args).
- `is_dml_device()` truth table.
- Forward-pass gating: with a stubbed device type, assert the DML branch moves
  complex ops to CPU and the MPS/CUDA/CPU branches are untouched (structure
  tests around the helper; full-tensor DML tests aren't possible off-Windows).
- Regression guard: MPS still hops **only** at istft.

## Rollout / PR sequencing

Cross-repo: infra PRs land in **karaoke-gen** (dispatcher + image bake),
code/workflow PRs land in **python-audio-separator**.

1. **PR 1 (audio-separator) — Windows CPU integration job** (CI only, no code
   change). Green baseline proves Windows e2e before we change behavior.
   Include the `runs-on: [self-hosted, linux, gpu]` label-routing fix for the
   three existing jobs in this PR (safe no-op today, prerequisite for Windows
   runners existing at all).
2. **PR 2 (karaoke-gen) — Windows runner infra**: `gpu-windows` family in
   `ephemeral.py` (+ `test_ephemeral.py`), PowerShell startup script,
   `runner-image-provision.ps1`, `build-runner-images.yml` variant. Then
   manually dispatch the bake workflow to produce the first
   `gha-runner-gpu-windows` image and verify a throwaway job runs on it.
   (Remember: `pulumi up` locally before merge per karaoke-gen conventions if
   any Pulumi-managed resources change — the dispatcher function redeploys.)
3. **PR 3 (audio-separator) — loader `map_location` fix** + unit tests +
   `windows-directml` job as `continue-on-error`. Real DML feedback per push
   from here on.
4. **PR 4 (audio-separator) — complex-op CPU hops** in both roformer models
   (+ attention fallback if the DML run reveals SDPA failure) + unit tests.
5. **PR 5 (audio-separator) — promote to required**: move `windows-directml`
   (and `windows-cpu-integration`) into the gate job + add to ruleset 529535
   + README DirectML compatibility table update (RoFormer ✅, Demucs ❌ with
   reason) + docs/CI-GPU-RUNNERS.md update for the Windows family.
6. **Community validation**: comment on #292 with a branch/pre-release wheel
   and ask Vageesha-Gupta to re-run their matrix on the AMD iGPU; their
   results are the AMD sign-off our NVIDIA CI can't provide.

## Regression safety net (why other platforms can't break)

- Every behavioral change is behind `device.type == "privateuseone"`, which is
  unreachable unless `use_directml=True` **and** torch_directml is installed
  (Windows-only optional extra).
- Existing Linux GPU integration suite (3 jobs, CUDA) must stay green — covers
  the roformer files we're editing on the primary platform.
- Existing macOS unit tests + the MPS-specific assertions cover the MPS branch.
- New Windows CPU integration job covers the CPU-on-Windows path.

## Decisions (Andrew, 2026-07-19)

1. **Runner infra**: extend the existing ephemeral self-hosted runner fleet
   (karaoke-gen runner_manager + image bake) with a Windows GPU family —
   NOT GitHub-hosted larger runners.
2. **Required checks**: the new Windows jobs join the existing mandatory
   integration-test checks before PR merge (ruleset 529535), after a
   stabilization period running as `continue-on-error`.

## Implementation-time verification list

- torch-directml Python ceiling (expect 3.12; no 3.13 wheels as of writing).
- GRID driver installer path in `gs://nvidia-drivers-us-public` for Windows
  Server 2022 + T4, and that `dxdiag`/torch_directml sees a WDDM DX12 adapter.
- `model.to("privateuseone:0")` from a plain string (confirmed working via the
  reporter's legacy-path log, but assert it in the DML CI job anyway).
- Whether SDPA works on torch-directml (decides if Phase 2c is needed).
