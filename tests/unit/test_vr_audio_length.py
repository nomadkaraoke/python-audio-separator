"""Audio boundary contracts for VR separation."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from audio_separator.separator.architectures.vr_separator import VRSeparator
from audio_separator.separator.uvr_lib_v5 import spec_utils


SAMPLE_RATE = 44100


@pytest.fixture(autouse=True)
def use_portable_vr_resampler(monkeypatch):
    """Keep boundary tests independent of the optional libsamplerate backend."""
    monkeypatch.setattr(spec_utils, "wav_resolution", "polyphase")


def _make_vr_separator(tmp_path, input_audio, vr_model_param="4band_v2", high_end_process=False):
    input_path = tmp_path / "input.wav"
    sf.write(input_path, input_audio, SAMPLE_RATE, subtype="PCM_16")

    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"\0")

    output_dir = tmp_path / "output"
    common_config = {
        "logger": logging.getLogger(__name__),
        "log_level": logging.INFO,
        "torch_device": torch.device("cpu"),
        "torch_device_cpu": torch.device("cpu"),
        "torch_device_mps": None,
        "onnx_execution_provider": [],
        "model_name": "test-vr",
        "model_path": str(model_path),
        "model_data": {"vr_model_param": vr_model_param, "primary_stem": "Vocals"},
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "output_bitrate": None,
        "normalization_threshold": 1.0,
        "amplification_threshold": 0.0,
        "enable_denoise": False,
        "output_single_stem": "Vocals",
        "invert_using_spec": False,
        "sample_rate": SAMPLE_RATE,
        "use_soundfile": True,
    }
    separator = VRSeparator(common_config, {"high_end_process": high_end_process})
    separator._ensure_model_loaded = Mock()
    separator.inference_vr = lambda mix, _device, _aggressiveness: (mix, mix)

    return separator, input_path, output_dir


@pytest.mark.parametrize(
    ("input_frames", "high_end_process"),
    [(4080, False), (4080, True)],
    ids=["partial-hop", "high-end-process"],
)
def test_vr_separation_preserves_input_length(tmp_path, input_frames, high_end_process):
    input_audio = np.zeros((input_frames, 2), dtype=np.float32)
    separator, input_path, output_dir = _make_vr_separator(tmp_path, input_audio, high_end_process=high_end_process)

    output_files = separator.separate(str(input_path))

    output_info = sf.info(output_dir / output_files[0])
    assert output_info.frames == input_frames


def test_vr_separation_preserves_input_length_after_output_resampling(tmp_path):
    input_frames = 4080
    input_audio = np.zeros((input_frames, 2), dtype=np.float32)
    separator, input_path, output_dir = _make_vr_separator(tmp_path, input_audio, vr_model_param="1band_sr32000_hl512")

    output_files = separator.separate(str(input_path))

    output_info = sf.info(output_dir / output_files[0])
    assert output_info.frames == input_frames


def test_vr_separation_reconstructs_audio_after_last_hop_boundary(tmp_path):
    input_frames = 4081
    tail_frames = 240
    tail_time = np.arange(tail_frames) / SAMPLE_RATE
    tail_signal = 0.5 * np.sin(2 * np.pi * 440 * tail_time)
    input_audio = np.zeros((input_frames, 2), dtype=np.float32)
    input_audio[-tail_frames:] = tail_signal[:, np.newaxis]
    separator, input_path, output_dir = _make_vr_separator(tmp_path, input_audio)

    output_files = separator.separate(str(input_path))

    output_audio, _ = sf.read(output_dir / output_files[0], dtype="float32", always_2d=True)
    output_tail = output_audio[input_frames - tail_frames :]
    assert output_tail.shape == (tail_frames, 2) and np.any(np.abs(output_tail) > 1e-4)


def test_vr_separation_matches_explicit_final_hop_padding(tmp_path):
    """Match a stable reference that explicitly pads the incomplete final hop.

    For ``3band_44100``, half the FFT window is shorter than one hop. Without a
    frame at the next hop boundary, extending ISTFT to the input length makes
    the right edge depend on a near-zero window sum and can create a large spike.
    The explicitly padded input supplies that frame independently; automatic
    boundary handling should reproduce the same first ``input_frames`` samples.
    """
    hop_length = 512
    input_frames = hop_length * 20 + hop_length - 1
    time = np.arange(input_frames) / SAMPLE_RATE
    # A high-band tone exposes the unstable edge after the model's band filters.
    signal = 0.5 * np.sin(2 * np.pi * 18000 * time)
    input_audio = np.column_stack([signal, signal]).astype(np.float32)

    automatic_dir = tmp_path / "automatic"
    automatic_dir.mkdir()
    automatic_separator, automatic_input, _ = _make_vr_separator(automatic_dir, input_audio, vr_model_param="3band_44100")
    automatic_separator.final_process = Mock()
    automatic_separator.separate(str(automatic_input))
    automatic_output = automatic_separator.final_process.call_args.args[1]

    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    explicit_padding = hop_length - input_frames % hop_length
    explicitly_padded_audio = np.pad(input_audio, ((0, explicit_padding), (0, 0)))
    explicit_separator, explicit_input, _ = _make_vr_separator(explicit_dir, explicitly_padded_audio, vr_model_param="3band_44100")
    explicit_separator.final_process = Mock()
    explicit_separator.separate(str(explicit_input))
    explicit_output = explicit_separator.final_process.call_args.args[1]

    np.testing.assert_array_equal(automatic_output, explicit_output[:input_frames])
