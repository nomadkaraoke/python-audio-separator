"""MDX window batching, device transfers, and per-file state contracts."""

import subprocess
import sys
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest
import torch

from audio_separator.separator.architectures.mdx_separator import MDXSeparator

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_separator(tmp_path, batch_dimension="batch", batch_size=3, device="cpu", segment_size=16, providers=None):
    """Create a real ONNX test graph and configure a separator without downloading weights."""
    shape = [batch_dimension, 4, 17, 16]
    graph = helper.make_graph(
        [helper.make_node("Neg", ["spectrogram"], ["prediction"])],
        "mdx-test",
        [helper.make_tensor_value_info("spectrogram", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "mdx.onnx"
    onnx.save(model, str(path))
    config = {
        "logger": logging.getLogger(__name__),
        "log_level": logging.INFO,
        "torch_device": torch.device(device),
        "torch_device_cpu": torch.device("cpu"),
        "onnx_execution_provider": providers or ["CPUExecutionProvider"],
        "model_path": str(path),
        "model_name": "mdx-test",
        "model_data": {"primary_stem": "Vocals", "compensate": 1.0, "mdx_dim_f_set": 17, "mdx_dim_t_set": 4, "mdx_n_fft_scale_set": 32},
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "sample_rate": 44100,
        "output_format": "WAV",
    }
    params = {"segment_size": segment_size, "hop_length": 8, "overlap": 0.25, "batch_size": batch_size, "enable_denoise": False}
    return MDXSeparator(config, params)


@pytest.mark.parametrize("batch_dimension", [1, 2, "batch", None])
def test_onnx_batch_metadata_and_input_names(tmp_path, batch_dimension):
    """Honor declared batch dimensions and resolve nonstandard ONNX input names."""
    separator = make_separator(tmp_path, batch_dimension=batch_dimension)
    batch = batch_dimension if isinstance(batch_dimension, int) else 3
    spectrum = torch.randn(batch, 4, 17, 16)
    np.testing.assert_array_equal(separator.model_run(spectrum), -spectrum.numpy())
    assert separator.inference_batch_size == batch
    assert separator.fixed_batch_size == (batch_dimension if isinstance(batch_dimension, int) else None)


@pytest.mark.parametrize("overlap", [0, 0.25, 0.75, 0.99])
@pytest.mark.parametrize("length", [1, 1009])
@pytest.mark.parametrize("fixed_batch", [None, 2])
def test_batched_demix_preserves_samples_and_pads_last_batch(tmp_path, overlap, length, fixed_batch):
    """Preserve overlap-add samples and honor batch limits for short and partial windows."""
    separator = make_separator(tmp_path, batch_dimension=fixed_batch or "batch")
    mix = np.random.default_rng(42).normal(0, 0.1, (2, length)).astype(np.float32)
    separator.overlap = overlap
    calls = []

    def predict(batch, is_match_mix=False):
        """Record the submitted batch size and return deterministic scaled audio."""
        calls.append(batch.shape[0])
        return batch.cpu().numpy() * 0.3

    separator.run_model = predict
    actual = separator.demix(mix)
    assert np.isfinite(actual).all()
    assert actual.shape == mix.shape
    np.testing.assert_allclose(actual, mix * 0.3, atol=1e-7)
    if fixed_batch:
        assert set(calls) == {fixed_batch}
    else:
        assert all(1 <= batch <= 3 for batch in calls)
        if length > separator.chunk_size:
            assert max(calls) == 3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_dimension", [1, 2, "batch"])
def test_real_onnx_demix_matches_single_window_batches(tmp_path, device, batch_dimension):
    """Match serial-window reconstruction when real ONNX inference runs in batches."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" and CUDA_ONNX_AVAILABLE else ["CPUExecutionProvider"]
    separator = make_separator(tmp_path, batch_dimension=batch_dimension, device=device, providers=providers)
    mix = np.random.default_rng(42).normal(0, 0.1, (2, 1009)).astype(np.float32)
    batched = separator.demix(mix)
    separator.inference_batch_size = separator.fixed_batch_size or 1
    np.testing.assert_allclose(separator.demix(mix), batched, atol=2e-7)


@pytest.mark.parametrize("denoise", [False, True])
@pytest.mark.parametrize("match_mix", [False, True])
def test_torch_predictions_remain_on_device(tmp_path, denoise, match_mix):
    """Preserve prediction values and tensor identity through the inverse STFT boundary."""
    separator = make_separator(tmp_path)
    separator.enable_denoise = denoise
    prediction = torch.randn(2, 4, 17, 16)
    separator.stft = Mock(return_value=prediction.clone())
    predictions = []

    def predict(value):
        """Retain each generated prediction so the test can detect unnecessary tensor copies."""
        result = value * 0.3
        predictions.append(result)
        return result

    separator.model_run = Mock(side_effect=predict)
    seen = []

    def inverse(value):
        """Capture the prediction received by inverse STFT and return a valid audio-shaped tensor."""
        seen.append(value)
        return torch.zeros(2, 2, 120)

    separator.stft.inverse.side_effect = inverse
    separator.run_model(torch.zeros(2, 2, 120), is_match_mix=match_mix)
    expected = prediction.clone()
    expected[:, :, :3] = 0
    if not match_mix:
        expected *= 0.3
    torch.testing.assert_close(seen[0], expected)
    assert separator.model_run.call_count == (0 if match_mix else 2 if denoise else 1)
    if match_mix:
        assert seen[0] is separator.stft.return_value
    elif not denoise:
        assert seen[0] is predictions[0]


@pytest.mark.parametrize(
    "selection,expected_names", [(None, ["Instrumental", "Vocals"]), ("vocals", ["Vocals"]), ("instrumental", ["Instrumental"])]
)
@pytest.mark.parametrize("invert", [False, True])
def test_secondary_reconstruction_only_runs_when_needed(tmp_path, monkeypatch, selection, expected_names, invert):
    """Run mixture reconstruction only when a requested secondary stem needs spectral inversion."""
    separator = make_separator(tmp_path)
    separator.output_single_stem = selection
    separator.invert_using_spec = invert
    separator.prepare_mix = Mock(return_value=np.ones((2, 32), dtype=np.float32) * 0.2)
    separator.demix = Mock(side_effect=lambda mix, is_match_mix=False: mix * (1 if is_match_mix else 0.3))
    separator.get_stem_output_path = Mock(side_effect=lambda name, custom: name + ".wav")
    separator.final_process = Mock()
    invert_stem = Mock(side_effect=lambda raw, primary: raw.T - primary)
    monkeypatch.setattr("audio_separator.separator.architectures.mdx_separator.spec_utils.invert_stem", invert_stem)
    assert separator.separate("input.wav") == [name + ".wav" for name in expected_names]
    needs_match = invert and selection != "vocals"
    assert separator.demix.call_count == (2 if needs_match else 1)
    assert invert_stem.call_count == int(needs_match)


def test_reused_separator_does_not_export_previous_audio(tmp_path):
    """Replace cached sources when one separator processes successive input files."""
    separator = make_separator(tmp_path)
    separator.prepare_mix = Mock(side_effect=[np.ones((2, 32), dtype=np.float32) * level for level in (0.1, 0.2)])
    separator.demix = Mock(side_effect=lambda mix: mix * 0.3)
    separator.get_stem_output_path = Mock(side_effect=lambda name, custom: name + ".wav")
    saved = []
    separator.final_process = lambda path, source, name: saved.append((name, source.copy()))
    separator.separate("first.wav")
    separator.separate("second.wav")
    assert [name for name, _ in saved] == ["Instrumental", "Vocals"] * 2
    assert not np.array_equal(saved[0][1], saved[2][1])
    assert not np.array_equal(saved[1][1], saved[3][1])


def test_native_module_does_not_import_torch_converter():
    """Keep native MDX imports independent of the optional Torch conversion module."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import audio_separator.separator.architectures.mdx_separator; assert 'onnx2torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_converted_model_keeps_requested_batch_size(tmp_path, monkeypatch):
    """Use the requested window batch size when ONNX conversion produces a Torch model."""
    import sys

    model = torch.nn.Identity()
    monkeypatch.setitem(sys.modules, "onnx2torch", SimpleNamespace(convert=Mock(return_value=model)))
    separator = make_separator(tmp_path, segment_size=32, batch_size=3)
    assert separator.uses_pytorch_inference
    assert separator.fixed_batch_size is None
    assert separator.inference_batch_size == 3
    assert separator.model_run is model


def test_invalid_chunk_size_fails_before_allocating_windows(tmp_path):
    """Reject segment geometry that cannot accommodate the FFT trimming window."""
    separator = make_separator(tmp_path)
    separator.segment_size = 2
    with pytest.raises(ValueError, match="FFT window"):
        separator.demix(np.zeros((2, 32), dtype=np.float32))


CUDA_ONNX_AVAILABLE = torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()


@pytest.mark.skipif(not CUDA_ONNX_AVAILABLE, reason="CUDA Torch and ONNX Runtime are required")
@pytest.mark.parametrize("batch_dimension", [1, 2, "batch"])
def test_cuda_binding_handles_noncontiguous_inputs_and_nondefault_streams(tmp_path, batch_dimension):
    """Synchronize CUDA buffers across streams and keep results independent between calls."""
    separator = make_separator(
        tmp_path,
        device="cuda",
        batch_dimension=batch_dimension,
        providers=[("CUDAExecutionProvider", {"do_copy_in_default_stream": True}), "CPUExecutionProvider"],
    )
    assert "CUDAExecutionProvider" in separator.onnx_session.get_providers()
    assert separator.onnx_stream is not None
    assert separator.onnx_session.get_provider_options()["CUDAExecutionProvider"]["has_user_compute_stream"] == "1"
    batch_sizes = [1, 3] if batch_dimension == "batch" else [batch_dimension]
    for batch in batch_sizes:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            spectrum = torch.randn(batch, 4, 16, 17, device="cuda").transpose(-1, -2)
            actual = separator.model_run(spectrum)
            assert actual.device == spectrum.device
            torch.testing.assert_close(actual, -spectrum, rtol=0, atol=0)
            saved = actual.clone()
            separator.model_run(spectrum * 2)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        stream.synchronize()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("output_shape", [None, [None, 4, 17, 16], ["unknown", 4, 17, 16]])
def test_unknown_output_shapes_use_allocating_fallback(tmp_path, device, output_shape):
    """Use NumPy allocation when output metadata cannot establish safe buffer dimensions."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" and CUDA_ONNX_AVAILABLE else ["CPUExecutionProvider"]
    separator = make_separator(tmp_path, device=device, providers=providers)
    metadata = separator.onnx_output
    separator.onnx_output = SimpleNamespace(name=metadata.name, type=metadata.type, shape=output_shape)
    spectrum = torch.randn(3, 4, 17, 16, device=device)
    result = separator.model_run(spectrum)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, -spectrum.cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Torch is required")
def test_cuda_provider_failure_retains_numpy_fallback(tmp_path, monkeypatch, caplog):
    """Retain correct inference and a warning when the requested CUDA provider is unavailable."""
    session_factory = ort.InferenceSession

    def cpu_session(*args, **kwargs):
        """Force a real CPU session to simulate CUDA provider activation failure."""
        kwargs["providers"] = ["CPUExecutionProvider"]
        return session_factory(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", cpu_session)
    separator = make_separator(tmp_path, device="cuda", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert separator.onnx_stream is None
    spectrum = torch.randn(3, 4, 17, 16, device="cuda")
    result = separator.model_run(spectrum)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, -spectrum.cpu().numpy())
    assert "could not activate requested provider CUDAExecutionProvider" in caplog.text


@pytest.mark.parametrize("overlap", [-0.1, 1.0, float("inf"), float("nan")])
def test_invalid_overlap_fails_before_scheduling_windows(tmp_path, overlap):
    """Reject invalid overlap values before they can schedule excessive or invalid window work."""
    separator = make_separator(tmp_path)
    separator.overlap = overlap
    with pytest.raises(ValueError, match="MDX overlap"):
        separator.demix(np.zeros((2, 32), dtype=np.float32))


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_size_is_rejected(tmp_path, batch_size):
    """Reject batch sizes that cannot advance the window-processing loop."""
    separator = make_separator(tmp_path, batch_size=batch_size)
    with pytest.raises(ValueError, match="MDX batch size"):
        separator.demix(np.zeros((2, 32), dtype=np.float32))


def test_native_separator_releases_without_waiting_for_cycle_collection(tmp_path):
    """Release native separator ownership immediately without relying on cyclic garbage collection."""
    import gc
    import weakref

    separator = make_separator(tmp_path)
    reference = weakref.ref(separator)
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        del separator
        assert reference() is None
    finally:
        if collection_was_enabled:
            gc.enable()
