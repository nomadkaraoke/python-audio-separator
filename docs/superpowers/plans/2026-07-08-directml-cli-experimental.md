# Expose DirectML via CLI (experimental tier) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing, working DirectML acceleration reachable from the CLI via an explicit `--use_directml` flag, add a targeted discoverability hint, document it honestly as experimental, and prepare a patch release.

**Architecture:** DirectML support already exists end-to-end in `Separator` (PR #211) but is only reachable from the Python API. We wire it to the CLI (mirroring the existing `--use_autocast` flag), add one conditional INFO hint in the CPU-fallback path of `setup_torch_device()`, document a new experimental install section, and bump the patch version. No runtime/inference code changes.

**Tech Stack:** Python 3.10+, argparse, pytest, Poetry.

## Global Constraints

- **Support tier:** experimental / best-effort. All user-facing copy must say "experimental".
- **Explicit opt-in only** — DirectML must NOT auto-enable; it stays behind `use_directml` / `--use_directml`.
- **No behavior change for CUDA / MPS / CPU users** — the DirectML branch is only reachable when CUDA and MPS are both unavailable AND `use_directml=True` AND `torch_directml` is installed & available.
- **No remote CLI / API / deploy-server changes** — DirectML is local-only; do not touch `audio_separator/remote/*` or `tests/unit/test_remote_cli.py`.
- **No DirectML execution tests** — DML cannot run in CI; test only wiring and the hint.
- **Confirmed-working architecture:** MDX (`.onnx`) only. MDXC/roformer & VR expected but community-untested; Demucs unverified. Docs must state this.
- Current version: `0.44.2` → bump to `0.44.3`.

---

### Task 1: Expose `--use_directml` on the CLI and forward it to `Separator`

**Files:**
- Modify: `audio_separator/utils/cli.py` (help text ~line 63; argument ~line 78; forwarding ~line 249)
- Test: `tests/unit/test_cli.py` (fixture line 43; new test after `test_cli_use_autocast_argument`, ~line 258)

**Interfaces:**
- Consumes: existing `Separator(use_directml: bool = False, ...)` constructor param (`audio_separator/separator/separator.py:123`).
- Produces: CLI arg `args.use_directml` (bool, default `False`), forwarded as `use_directml=args.use_directml` in the `Separator(...)` call.

**Coupling note:** As soon as `cli.py` forwards `use_directml`, every existing test that asserts `assert_called_once_with(**common_expected_args)` will fail unless the fixture gains the key. The fixture change and the `cli.py` forwarding MUST land in this same task/commit.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_cli.py`, immediately after `test_cli_use_autocast_argument` (after line 258):

```python
# Test using use_directml argument
def test_cli_use_directml_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_directml"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["use_directml"] = True

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_cli.py::test_cli_use_directml_argument -v`
Expected: FAIL — argparse rejects the unknown flag, raising `SystemExit: 2` ("unrecognized arguments: --use_directml").

- [ ] **Step 3: Add the CLI argument, help text, forwarding, and fixture key**

In `audio_separator/utils/cli.py`, add the help variable after line 63 (`use_autocast_help = ...`):

```python
    use_directml_help = "Use DirectML for hardware-accelerated inference on Windows AMD/Intel GPUs (experimental; requires the 'dml' extra). Example: --use_directml"
```

Add the argument after line 78 (`common_params.add_argument("--use_autocast", ...)`):

```python
    common_params.add_argument("--use_directml", action="store_true", help=use_directml_help)
```

Forward it in the `Separator(...)` construction, immediately after line 249 (`use_autocast=args.use_autocast,`):

```python
        use_directml=args.use_directml,
```

In `tests/unit/test_cli.py`, add to the `common_expected_args` fixture (after line 43, `"use_autocast": False,`):

```python
        "use_directml": False,
```

- [ ] **Step 4: Run the CLI tests to verify all pass**

Run: `python -m pytest tests/unit/test_cli.py -v`
Expected: PASS — the new `test_cli_use_directml_argument` passes, and all existing tests that use `common_expected_args` still pass (fixture now carries `use_directml=False`, matching the new forwarded arg).

- [ ] **Step 5: Commit**

```bash
git add audio_separator/utils/cli.py tests/unit/test_cli.py
git commit -m "feat: expose --use_directml CLI flag (experimental)"
```

---

### Task 2: Add conditional DirectML discoverability hint

**Files:**
- Modify: `audio_separator/separator/separator.py` (CPU-fallback block in `setup_torch_device`, lines 399-402)
- Test: `tests/unit/test_directml.py` (new file)

**Interfaces:**
- Consumes: `self.use_directml` (bool), `self.get_package_distribution(name)` → distribution or `None`, and the local `has_torch_dml_installed` var already computed at `separator.py:383`.
- Produces: one INFO log line when DirectML packages are installed but DirectML is not enabled. No return value, no state change.

**Behavior note:** `audio-separator --env_info` constructs a full `Separator()` (not `info_only`), so this hint fires during `--env_info` on a DirectML-capable machine with no CUDA/MPS — the intended discovery path.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_directml.py`:

```python
import logging
import platform
from unittest.mock import MagicMock, patch

from audio_separator.separator import Separator

HINT = "DirectML packages detected but DirectML is not enabled"


def _run_setup(use_directml, dml_installed):
    """Construct a Separator without auto device-setup, then drive setup_torch_device
    with CUDA and MPS forced unavailable so the CPU-fallback path always runs."""
    sep = Separator(info_only=True)
    sep.use_directml = use_directml

    def fake_dist(name):
        if name in ("torch_directml", "onnxruntime-directml") and dml_installed:
            return MagicMock()
        return None

    with patch.object(sep, "get_package_distribution", side_effect=fake_dist), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False), \
         patch("audio_separator.separator.separator.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
        sep.setup_torch_device(platform.uname())
    return sep


def test_directml_hint_shown_when_packages_present_but_disabled(caplog):
    with caplog.at_level(logging.INFO):
        _run_setup(use_directml=False, dml_installed=True)
    assert any(HINT in r.message for r in caplog.records)


def test_directml_hint_absent_when_no_packages(caplog):
    with caplog.at_level(logging.INFO):
        _run_setup(use_directml=False, dml_installed=False)
    assert not any(HINT in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_directml.py -v`
Expected: `test_directml_hint_shown_when_packages_present_but_disabled` FAILS (hint not emitted yet); `test_directml_hint_absent_when_no_packages` PASSES (no hint exists to leak).

- [ ] **Step 3: Implement the hint**

In `audio_separator/separator/separator.py`, replace the CPU-fallback block (currently lines 399-402):

```python
        if not hardware_acceleration_enabled:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]
```

with:

```python
        if not hardware_acceleration_enabled:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

            # Discoverability hint: DirectML is an explicit opt-in (experimental). If the
            # DirectML packages are installed but the feature wasn't enabled, tell the user how.
            if not self.use_directml and (has_torch_dml_installed or self.get_package_distribution("onnxruntime-directml") is not None):
                self.logger.info(
                    "DirectML packages detected but DirectML is not enabled. "
                    "Pass use_directml=True (or --use_directml on the CLI) to enable experimental DirectML acceleration."
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_directml.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_separator/separator/separator.py tests/unit/test_directml.py
git commit -m "feat: hint that --use_directml is available when DML packages are installed"
```

---

### Task 3: Document DirectML as an experimental install option

**Files:**
- Modify: `README.md` (add a new install section after the CPU section; the CUDA section starts at line 93, Apple Silicon at line 115)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the experimental DirectML install section**

In `README.md`, add a new section alongside the other hardware install sections (place it after the CPU install section, before `## Usage`). Match the existing heading style:

```markdown
### 🪟 Windows AMD / Intel GPU with DirectML (experimental)

> **Experimental / community-supported.** DirectML acceleration was contributed by the community and is not tested in CI or by the maintainer (it requires Windows plus an AMD or Intel GPU). It is opt-in and will never affect CUDA, Apple Silicon, or CPU users.

Install with the `dml` extra:

```sh
pip install "audio-separator[dml]"
```

Then enable it explicitly with the `--use_directml` flag:

```sh
audio-separator path/to/audio.wav --use_directml
```

💬 If DirectML is configured correctly you will see this log line when running `audio-separator --env_info`:
`ONNXruntime has DmlExecutionProvider available, enabling acceleration`

**Model architecture status on DirectML:**

| Architecture | Model types | Status |
|---|---|---|
| MDX | `.onnx` | ✅ Confirmed working |
| MDXC (incl. the default `bs_roformer` model) | `.ckpt` / `.yaml` | ⚠️ Expected to work, community-untested |
| VR | `.pth` | ⚠️ Expected to work, community-untested |
| Demucs | — | ❓ Unverified |

If you test any of the untested architectures, please [open an issue](https://github.com/nomadkaraoke/python-audio-separator/issues) with your `--env_info` output and logs — reports are what move these from "untested" to "confirmed".
```

- [ ] **Step 2: Verify the README renders cleanly**

Run: `python -c "import pathlib; t = pathlib.Path('README.md').read_text(); assert '--use_directml' in t and 'DirectML' in t and 'experimental' in t.lower(); print('README OK')"`
Expected: prints `README OK`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add experimental DirectML (Windows AMD/Intel GPU) install section"
```

---

### Task 4: Bump patch version

**Files:**
- Modify: `pyproject.toml` (line 7)

**Interfaces:** none.

- [ ] **Step 1: Bump the version**

In `pyproject.toml`, change line 7 from:

```toml
version = "0.44.2"
```

to:

```toml
version = "0.44.3"
```

- [ ] **Step 2: Verify**

Run: `grep '^version' pyproject.toml`
Expected: `version = "0.44.3"`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.44.3"
```

---

### Task 5: Draft the reply to Vageesha

**Files:**
- Create: reply draft in the session scratchpad (NOT committed to the repo — it's an email draft, not project content).

**Interfaces:** none.

- [ ] **Step 1: Write the reply draft**

Write to `<scratchpad>/vageesha-directml-reply.md` and present it inline to the user for sending. Content must cover:
  1. Confirm DirectML is a real, working feature (PR #211) that was only ever exposed via the Python API — the missing CLI wiring was an oversight, not an intentional disable. Her diagnosis and workaround were exactly right.
  2. State that `--use_directml` is landing in v0.44.3 (installable via `pip install "audio-separator[dml]"`), plus the new `--env_info` discoverability hint.
  3. Thank her for the precise write-up.
  4. Light validation invite: since she offered logs, ask if she'd try MDXC/roformer, VR, and Demucs models on her AMD GPU and report which work — noting MDX is the only architecture confirmed so far. No pressure, no formal commitment.

- [ ] **Step 2: Present the draft to the user** (no commit).

---

## Final verification (after all tasks)

- [ ] Run the full unit suite: `python -m pytest tests/unit/ -q` — expect all green.
- [ ] Confirm no `audio_separator/remote/*` files or `tests/unit/test_remote_cli.py` were modified: `git diff --name-only origin/main` should list only `cli.py`, `separator.py`, `test_cli.py`, `test_directml.py`, `README.md`, `pyproject.toml`, and the docs under `docs/superpowers/`.

## Handoff to release

After implementation, ship via the standard workflow: `/test-review` → `/docs-review` → `/coderabbit` → `/pr` (adds `@coderabbitai ignore`) → merge → PyPI release via the existing `publish-to-pypi` workflow. Then send the reply from Task 5.
