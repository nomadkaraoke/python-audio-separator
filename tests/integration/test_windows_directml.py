"""Windows DirectML integration tests (issue #292).

Runs real separations with --use_directml on a Windows machine with a
DirectX 12 GPU (CI: the self-hosted gha-runner-gpu-windows ephemeral fleet,
NVIDIA T4 in WDDM mode). Asserts, per DML-supported architecture:

  * the separation completes and produces non-empty, non-silent, finite audio
    (DirectML producing silent garbage must fail, even if it "succeeds"),
  * output quality matches the committed reference images (same waveform/
    spectrogram similarity thresholds as the Linux CUDA integration tests),
  * RoFormer models load via the NEW implementation with no silent fallback
    to legacy (the loader map_location fix regression guard).

Skipped automatically when torch-directml isn't installed, so it never runs
in the Linux/macOS/Windows-CPU jobs.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

# Skip BEFORE importing test_cli_integration (it pulls in skimage/matplotlib
# via tests/utils) so DML-less environments skip cleanly at collection time.
torch_directml = pytest.importorskip("torch_directml")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # tests/ (for utils)
sys.path.append(os.path.dirname(__file__))  # tests/integration/
from test_cli_integration import resolve_cli_executable, validate_audio_output

pytestmark = pytest.mark.skipif(
    not torch_directml.is_available(), reason="torch-directml reports no DirectML device"
)

INPUT_FILE = "tests/inputs/mardy20s.flac"
REFERENCE_DIR = "tests/inputs/reference"

# torch-directml's allocator can't sustain the default MDXC segment size
# (801 for these roformer models) at fp32 — 'DML allocator out of memory' —
# so ckpt-based models run with a reduced segment. This is the documented
# resource knob for constrained GPUs, not a DML-specific hack.
DML_SEGMENT_ARGS = ["--mdxc_override_model_segment_size", "--mdxc_segment_size", "256"]

# (model, expected output files, is_roformer, validate_reference, extra_args)
DML_MODEL_PARAMS = [
    (
        # The exact model from the issue #292 report
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        [
            "mardy20s_(Instrumental)_model_bs_roformer_ep_317_sdr_12.flac",
            "mardy20s_(Vocals)_model_bs_roformer_ep_317_sdr_12.flac",
        ],
        True,
        True,
        DML_SEGMENT_ARGS,
    ),
    (
        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        [
            "mardy20s_(Instrumental)_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.flac",
            "mardy20s_(Vocals)_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.flac",
        ],
        True,
        True,
        DML_SEGMENT_ARGS,
    ),
    (
        # MDX — regression guard: already worked on DirectML before #292
        "UVR-MDX-NET-Inst_HQ_4.onnx",
        [
            "mardy20s_(Instrumental)_UVR-MDX-NET-Inst_HQ_4.flac",
            "mardy20s_(Vocals)_UVR-MDX-NET-Inst_HQ_4.flac",
        ],
        False,
        True,
        [],
    ),
    (
        # VR — regression guard: already worked on DirectML before #292
        "2_HP-UVR.pth",
        [
            "mardy20s_(Instrumental)_2_HP-UVR.flac",
            "mardy20s_(Vocals)_2_HP-UVR.flac",
        ],
        False,
        True,
        [],
    ),
    (
        # Plain MDXC (TFC_TDF arch) — its STFT wrapper already CPU-hops
        # non-cuda/cpu devices, so it should work on DML unchanged. No
        # committed reference images for this model: audibility/finiteness
        # checks only. Auto-downloads on first run (not in the baked set).
        "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        [
            "mardy20s_(Instrumental)_MDX23C-8KFFT-InstVoc_HQ.flac",
            "mardy20s_(Vocals)_MDX23C-8KFFT-InstVoc_HQ.flac",
        ],
        False,
        False,
        DML_SEGMENT_ARGS,
    ),
]

# Same defaults as test_cli_integration; DML numerics may drift slightly more
# than CUDA vs the CPU-generated references, revisit per-model if needed.
WAVEFORM_THRESHOLD = 0.90
SPECTROGRAM_THRESHOLD = 0.80

# Guards against "ran fine, produced silence" — well below any real stem,
# well above float noise.
RMS_FLOOR = 1e-4


def _assert_audible_and_finite(path):
    data, _sr = sf.read(path)
    assert np.isfinite(data).all(), f"{path} contains NaN/Inf samples"
    rms = float(np.sqrt(np.mean(np.square(data))))
    assert rms > RMS_FLOOR, f"{path} is (near-)silent: rms={rms:.2e}"


@pytest.mark.parametrize("model,expected_files,is_roformer,validate_reference,extra_args", DML_MODEL_PARAMS)
def test_dml_separation(model, expected_files, is_roformer, validate_reference, extra_args):
    for f in expected_files:
        if os.path.exists(f):
            os.remove(f)

    result = subprocess.run(
        [resolve_cli_executable(), "--use_directml", "--log_level", "debug", *extra_args, "-m", model, INPUT_FILE],
        capture_output=True,
        text=True,
        check=False,
    )
    log_text = (result.stdout or "") + (result.stderr or "")

    assert result.returncode == 0, f"CLI failed for {model}:\n{log_text[-4000:]}"

    # DirectML must actually be engaged — not silently fallen back to CPU.
    assert "DirectML is available in Torch, setting Torch device to DirectML" in log_text
    assert "ONNXruntime has DmlExecutionProvider available, enabling acceleration" in log_text

    if is_roformer:
        # Loader regression guard: the map_location fix means the NEW
        # implementation must load — a silent legacy fallback would keep CI
        # green while shipping the unfixed path.
        assert "Fell back to legacy" not in log_text, f"{model} silently fell back to legacy implementation"
        assert "with new implementation" in log_text, f"{model} did not report new-implementation load"

    for output_file in expected_files:
        assert os.path.exists(output_file), f"Output file {output_file} was not created"
        assert os.path.getsize(output_file) > 0, f"Output file {output_file} is empty"
        _assert_audible_and_finite(output_file)

        if validate_reference and os.environ.get("SKIP_AUDIO_VALIDATION") != "1":
            waveform_match, spectrogram_match = validate_audio_output(
                output_file, REFERENCE_DIR, WAVEFORM_THRESHOLD, SPECTROGRAM_THRESHOLD
            )
            assert waveform_match, f"Waveform similarity below threshold for {output_file}"
            assert spectrogram_match, f"Spectrogram similarity below threshold for {output_file}"

    for f in expected_files:
        if os.path.exists(f):
            os.remove(f)
