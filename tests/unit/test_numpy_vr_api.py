import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from audio_separator.separator import Separator
from audio_separator.separator.architectures import vr_separator
from audio_separator.separator.architectures.vr_separator import VRSeparator
from audio_separator.separator.execution_policy import FP32
from audio_separator.separator.uvr_lib_v5.vr_network.model_param_init import ModelParameters

MODEL_PARAMS_DIR = Path(vr_separator.__file__).resolve().parents[1] / "uvr_lib_v5" / "vr_network" / "modelparams"
# VR rebuilds waveforms from STFT frames, so stems may end a few milliseconds early.
MAX_LENGTH_DIFF = int(0.05 * 44100)


def stereo_tone(frames=44100):
    t = np.arange(frames, dtype=np.float32) / 44100
    return np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 660 * t)], axis=1).astype(np.float32)


@pytest.fixture
def make_vr_numpy_separator(tmp_path, monkeypatch):
    for method in ("load", "get_duration"):
        monkeypatch.setattr(
            f"audio_separator.separator.common_separator.librosa.{method}",
            Mock(side_effect=AssertionError("Unexpected audio file read")),
        )
    monkeypatch.setattr(
        "audio_separator.separator.uvr_lib_v5.spec_utils.write_array_to_mem",
        Mock(side_effect=AssertionError("Unexpected in-memory WAV round trip")),
    )
    # Use scipy's polyphase resampler, as VR does on Apple Silicon, so these tests don't need libsamplerate.
    monkeypatch.setattr("audio_separator.separator.uvr_lib_v5.spec_utils.wav_resolution", "polyphase")
    weights = tmp_path / "test.pth"
    weights.write_bytes(b"\0")

    def make(param_name="1band_sr44100_hl512"):
        separator = Separator(model_file_dir=tmp_path / "models", output_dir=tmp_path / "output", info_only=True)
        model = VRSeparator.__new__(VRSeparator)
        model.logger = logging.getLogger(__name__)
        model.torch_device = torch.device("cpu")
        model.torch_device_mps = None
        model.effective_precision = FP32
        model.sample_rate = 44100
        model.normalization_threshold = 0.9
        model.amplification_threshold = 0.0
        model.output_single_stem = None
        model.input_bit_depth = None
        model.input_subtype = None
        model.primary_stem_name = "Instrumental"
        model.secondary_stem_name = "Vocals"
        model.model_name = "test"
        model.model_path = str(weights)
        model.model_params = ModelParameters(str(MODEL_PARAMS_DIR / f"{param_name}.json"))
        for band in model.model_params.param["band"].values():
            band["res_type"] = "polyphase"
        model.model_samplerate = model.model_params.param["sr"]
        model.is_vr_51_model = False
        model.high_end_process = False
        model.input_high_end_h = None
        model.input_high_end = None
        model.aggressiveness = {}
        model._ensure_model_loaded = Mock()
        # Only the network is mocked; band splitting, STFT and waveform rebuilding run for real.
        model.inference_vr = Mock(side_effect=lambda spec, device, aggressiveness: (spec * 0.25, spec * 0.75))
        model.clear_gpu_cache = Mock()
        model.write_audio = Mock(side_effect=AssertionError("Unexpected audio file write"))
        separator.torch_device = model.torch_device
        separator.model_instance = model
        separator.model_filename = "test.pth"
        separator.model_filenames = ["test.pth"]
        return separator

    return make


# 2band_32000 runs at 32 kHz, which covers the in-memory input and output resampling.
@pytest.mark.parametrize("param_name", ["1band_sr44100_hl512", "2band_32000"])
def test_vr_numpy_returns_stereo_arrays_without_audio_io(make_vr_numpy_separator, param_name):
    separator = make_vr_numpy_separator(param_name)
    audio = stereo_tone()
    audio.flags.writeable = False

    stems = separator.separate_numpy(audio, 44100)

    np.testing.assert_array_equal(audio, stereo_tone())
    assert set(stems) == {"Instrumental", "Vocals"}
    for stem in stems.values():
        assert stem.ndim == 2 and stem.shape[1] == 2
        assert abs(stem.shape[0] - audio.shape[0]) <= MAX_LENGTH_DIFF
        assert np.isfinite(stem).all()
        assert np.any(stem)
        assert not np.shares_memory(stem, audio)
    assert not np.shares_memory(stems["Instrumental"], stems["Vocals"])
    separator.model_instance.inference_vr.assert_called_once()


# A 128-frame clip is shorter than one STFT hop, so without padding the stems would come back empty.
@pytest.mark.parametrize("param_name", ["1band_sr44100_hl512", "2band_32000"])
def test_vr_numpy_keeps_sub_hop_input_length(make_vr_numpy_separator, param_name):
    separator = make_vr_numpy_separator(param_name)
    audio = stereo_tone(frames=128)

    stems = separator.separate_numpy(audio, 44100)

    assert set(stems) == {"Instrumental", "Vocals"}
    for stem in stems.values():
        assert stem.shape == (128, 2)
        assert np.isfinite(stem).all()
        assert np.any(stem)


def test_vr_numpy_converts_mono_to_stereo(make_vr_numpy_separator):
    separator = make_vr_numpy_separator()

    stems = separator.separate_numpy(stereo_tone()[:, 0], 44100)

    for stem in stems.values():
        assert stem.shape[1] == 2
        np.testing.assert_allclose(stem[:, 0], stem[:, 1], atol=1e-6)


def test_vr_numpy_honors_single_stem_selection(make_vr_numpy_separator):
    separator = make_vr_numpy_separator()
    separator.model_instance.output_single_stem = "vocals"

    stems = separator.separate_numpy(stereo_tone(), 44100)

    assert set(stems) == {"Vocals"}


def test_vr_file_mode_still_loads_and_writes_audio_files(make_vr_numpy_separator, monkeypatch, tmp_path):
    model = make_vr_numpy_separator().model_instance
    model.output_format = "FLAC"
    model.write_audio = Mock()
    load = Mock(return_value=(stereo_tone().T.copy(), 44100))
    monkeypatch.setattr("audio_separator.separator.architectures.vr_separator.librosa.load", load)
    monkeypatch.setattr("audio_separator.separator.uvr_lib_v5.spec_utils.write_array_to_mem", lambda path, subtype: path)
    input_path = str(tmp_path / "song.flac")

    output_files = model.separate(input_path)

    assert output_files == ["song_(Instrumental)_test.flac", "song_(Vocals)_test.flac"]
    assert load.call_args.args[0] == input_path
    assert model.write_audio.call_count == 2
