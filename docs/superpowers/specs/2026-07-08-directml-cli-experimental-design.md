# Expose DirectML via CLI (experimental tier) — Design

**Date:** 2026-07-08
**Status:** Approved (design)
**Author:** Andrew Beveridge (with Claude)
**Trigger:** Contributor email (Vageesha Gupta) reporting that DirectML support exists in the code but is unreachable from the CLI.

## Background

DirectML acceleration was contributed by an external contributor (zbear0808) in
PR #211 (May 2025). It is a real, working feature:

- `Separator.__init__` accepts `use_directml` (default `False`), stored as
  `self.use_directml` (`separator.py:123`, `:214`).
- `setup_torch_device()` branches into DirectML **only** when CUDA and MPS are
  both unavailable, `use_directml=True`, and `torch_directml` is installed and
  available (`separator.py:393-397`).
- `configure_dml()` sets the Torch device to DirectML and enables the ONNX
  Runtime `DmlExecutionProvider` (`separator.py:431-444`).
- Packaging already supports it: `pyproject.toml` defines a `dml` extra
  (`onnxruntime-directml` + `torch_directml`), and the README dev-setup mentions
  `poetry install --extras "dml"`.

**The gap:** `use_directml` is not exposed via the CLI. `cli.py` defines
`--use_soundfile` and `--use_autocast` and forwards them to `Separator(...)`, but
there is no `--use_directml` argument and it is not forwarded. A CLI user cannot
enable DirectML without editing source. This is an oversight in PR #211, not an
intentional gate.

A contributor (Vageesha Gupta) independently diagnosed this exactly, patched the
CLI locally, and confirmed `UVR-MDX-NET-Inst_HQ_3.onnx` runs with DirectML
hardware acceleration on a Windows 11 / AMD integrated GPU setup.

## Support posture (decision)

**Experimental / best-effort.** Expose the feature and document it honestly as
experimental and community-supported. The maintainer has no Windows/DirectML
machine and cannot test or CI this path, so we set expectations rather than
promise a support tier.

## Architecture compatibility (as understood today)

| Architecture | Model types | DirectML status |
|---|---|---|
| MDX | `.onnx` | **Confirmed working** (ONNX `DmlExecutionProvider`; contributor-tested). |
| MDXC (incl. `bs_roformer`, the default model) | `.ckpt` / `.yaml` | Patched in PR #211 (load on CPU, move to device). Expected to work; community-untested. |
| VR | `.pth` | Patched in PR #211 (load on CPU, move to device). Expected to work; community-untested. |
| Demucs | — | **Not touched** by PR #211; loads straight to device. Unverified. |

`use_autocast` self-disables on DirectML — `autocast_mode.is_autocast_available()`
returns `False` for the DML device type (`separator.py:1024`), so there is no
autocast/DML conflict to handle.

## Design decisions

**Fork 1 — Explicit opt-in (chosen) vs. auto-enable.**
Keep DirectML behind explicit `--use_directml` / `use_directml=True`. CUDA and MPS
auto-enable, but auto-enabling DirectML would silently route any user who merely
has `torch_directml` installed onto an untested acceleration path — an
unacceptable silent-regression risk for a feature we cannot test. Explicit opt-in
matches the experimental tier.

**Fork 2 — Conditional discoverability hint (chosen) vs. static env_info line.**
Emit a targeted INFO hint only when DirectML packages are installed but DirectML
is not active. Fires exactly for the users who would benefit; invisible to
everyone else.

## Scope

### 1. CLI wiring — `audio_separator/utils/cli.py`

- Add `--use_directml` to the "Common Separation Parameters" argument group,
  `action="store_true"`, mirroring `--use_autocast`.
  - Help text: *"Use DirectML for hardware-accelerated inference on Windows
    AMD/Intel GPUs (experimental; requires the `dml` extra). Example:
    --use_directml"*
- Forward `use_directml=args.use_directml` into the `Separator(...)` construction.

### 2. Conditional discoverability hint — `audio_separator/separator/separator.py`

- In `setup_torch_device()`, within the existing CPU-fallback block
  (`if not hardware_acceleration_enabled:`), if `torch_directml` **or**
  `onnxruntime-directml` is installed but `self.use_directml` is `False`, log one
  INFO line:
  *"DirectML packages detected but DirectML is not enabled. Pass
  `use_directml=True` (or `--use_directml` on the CLI) to enable experimental
  DirectML acceleration."*
- Reuses the existing `get_package_distribution(...)` and `has_torch_dml_installed`
  checks — no new detection logic.

### 3. Documentation — `README.md`

- Add an install section **"🪟 Windows AMD/Intel GPU with DirectML
  (experimental)"** mirroring the CUDA / Apple Silicon / CPU sections:
  - `pip install "audio-separator[dml]"`
  - The `--env_info` confirmation log line:
    `ONNXruntime has DmlExecutionProvider available, enabling acceleration`
  - The requirement to pass `--use_directml`.
- Include an honest status note reflecting the compatibility table above
  (MDX confirmed; MDXC/roformer & VR expected but untested; Demucs unverified),
  and ask users to open an issue with logs.
- Add `--use_directml` to the CLI usage/help reference block in the README if one
  is maintained there.

### 4. Tests — `tests/unit/test_cli.py`

- Add `"use_directml": False` to the `common_expected_args` fixture
  (`test_cli.py`). **Required:** once `cli.py` forwards `use_directml` into the
  `Separator(...)` call, every test that asserts
  `assert_called_once_with(**common_expected_args)` will fail unless the fixture
  includes the key. The fixture update and the `cli.py` forwarding must land
  together.
- Add `test_cli_use_directml_argument` mirroring `test_cli_use_autocast_argument`
  (asserts `--use_directml` results in `use_directml=True` in the `Separator`
  call).
- No DirectML *execution* test — DML cannot run in CI. We test only the
  CLI→constructor wiring, exactly as `use_autocast` is tested.
- **Remote CLI (`tests/unit/test_remote_cli.py`) is deliberately NOT touched** —
  see Non-goals.

### 5. Reply to contributor (draft; maintainer sends)

- Confirm DirectML is a real, working feature (PR #211) that was only ever exposed
  via the Python API — the CLI wiring was an oversight, not an intentional
  disable. Her workaround was correct.
- State that `--use_directml` is landing in the next patch release.
- Thank her for the precise diagnosis.
- Light validation invite: ask if she would test MDXC/roformer, VR, and Demucs
  models on her AMD GPU and report which work, noting MDX is the only architecture
  confirmed so far.

### 6. Release

- Patch version bump in `pyproject.toml`.
- PR (with `@coderabbitai ignore` per workflow), merge, PyPI release via the
  existing `publish-to-pypi` workflow.

## Non-goals (YAGNI)

- No auto-enable of DirectML — explicit opt-in only.
- No DirectML CI — infeasible (Windows + discrete non-NVIDIA GPU).
- No changes to Demucs or other architecture internals to "fix" DirectML — we
  document status; we do not chase untested fixes.
- No formal support commitment or ongoing compatibility-matrix maintenance
  burden.
- **No remote CLI / API / deploy-server changes.** DirectML is local-only
  (Windows + AMD/Intel GPU); the remote separation servers run on Cloud Run /
  Modal with CUDA or CPU and can never present a DirectML device. Plumbing
  `use_directml` through `audio_separator/remote/*` would be misleading dead
  configuration. The local `cli.py` change does not affect the remote tests
  (separate arg parser).

## Risk assessment

Very low. The DirectML branch is unreachable unless CUDA and MPS are both absent,
`torch_directml` is installed, and the user explicitly opts in. The CLI flag and
the conditional hint cannot change behavior for any CUDA, MPS, or CPU user. Tests
cover the wiring; the runtime path is unchanged from the already-merged PR #211.
