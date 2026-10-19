"""
GPU timing guard for RoFormer inference.

Separation time = forward passes x seconds per pass. tests/unit/test_mdxc_roformer_pass_budget.py
pins the pass count on every PR; this test measures the other factor on a real CUDA GPU, so a
change that makes each pass slower (precision, attention backend, device transfers, compile)
fails here instead of surfacing as slow production jobs.

The ceiling is deliberately generous (CI runs on a T4) and can be tuned with
ROFORMER_MAX_SECONDS_PER_PASS. The measured numbers are printed so drift is visible in the log.
"""

import os
import statistics
import time

import pytest
import torch

from audio_separator.separator import Separator

MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
INPUT = os.path.join(os.path.dirname(__file__), "..", "inputs", "mardy20s.flac")
MAX_SECONDS_PER_PASS = float(os.environ.get("ROFORMER_MAX_SECONDS_PER_PASS", "2.0"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="RoFormer timing guard needs a CUDA GPU")


def _time_separation(tmp_path, overlap):
    separator = Separator(output_dir=str(tmp_path), mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": None, "overlap": overlap, "pitch_shift": 0})
    separator.load_model(model_filename=MODEL)
    model_instance = separator.model_instance

    pass_seconds = []
    run_pass = model_instance._run_roformer_model

    def timed_pass(part):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = run_pass(part)
        torch.cuda.synchronize()
        pass_seconds.append(time.perf_counter() - started)
        return output

    model_instance._run_roformer_model = timed_pass
    separator.separate(INPUT)
    return model_instance.overlap, pass_seconds


@pytest.mark.parametrize(
    "overlap,expected_overlap,expected_passes",
    [
        (None, 4, 8),  # model YAML num_overlap
        (2, 2, 5),  # what karaoke-gen runs
    ],
)
def test_roformer_karaoke_seconds_per_pass(tmp_path, overlap, expected_overlap, expected_passes):
    effective_overlap, pass_seconds = _time_separation(tmp_path, overlap)

    assert effective_overlap == expected_overlap
    assert len(pass_seconds) == expected_passes

    # The first pass includes CUDA/cuDNN warm-up, so judge the steady state.
    steady = statistics.median(pass_seconds[1:])
    print(
        f"\n{MODEL} on {torch.cuda.get_device_name(0)} overlap={effective_overlap}: "
        f"{len(pass_seconds)} passes, first {pass_seconds[0]:.3f}s, median steady {steady:.3f}s, total {sum(pass_seconds):.2f}s"
    )
    assert steady <= MAX_SECONDS_PER_PASS, (
        f"Median RoFormer forward pass took {steady:.3f}s (> {MAX_SECONDS_PER_PASS}s). "
        "Per-pass inference got slower; check precision, attention backend and device transfers."
    )
