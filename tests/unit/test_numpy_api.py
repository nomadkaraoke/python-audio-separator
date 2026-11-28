import logging
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from audio_separator.separator import Separator
from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from audio_separator.separator.exceptions import InvalidAudioDataError
from audio_separator.separator.execution_policy import FP32


@pytest.fixture
def numpy_separator(tmp_path, monkeypatch):
    separator = Separator(model_file_dir=tmp_path / "models", output_dir=tmp_path / "output", info_only=True)
    model = MDXSeparator.__new__(MDXSeparator)
    model.logger = logging.getLogger(__name__)
    model.torch_device = torch.device("cpu")
    model.effective_precision = FP32
    model.sample_rate = 44100
    model.normalization_threshold = 0.9
    model.amplification_threshold = 0.0
    model.output_single_stem = None
    model.input_bit_depth = None
    model.input_subtype = None
    model.primary_stem_name = "Vocals"
    model.secondary_stem_name = "Instrumental"
    model.compensate = 1.0
    model.invert_using_spec = False
    model.demix = Mock(side_effect=lambda mix, is_match_mix=False: mix.copy())
    model.clear_gpu_cache = Mock()
    model.write_audio = Mock(side_effect=AssertionError("Unexpected audio file write"))
    for method in ("load", "get_duration"):
        monkeypatch.setattr(
            f"audio_separator.separator.common_separator.librosa.{method}",
            Mock(side_effect=AssertionError("Unexpected audio file read")),
        )
    separator.torch_device = model.torch_device
    separator.model_instance = model
    separator.model_filename = "test.onnx"
    separator.model_filenames = ["test.onnx"]
    return separator


def test_numpy_separation_preserves_read_only_input_and_avoids_audio_io(numpy_separator):
    audio = np.full((32, 2), 2.0, dtype=np.float32)
    audio.flags.writeable = False
    numpy_separator.chunk_duration = 0.001

    stems = numpy_separator.separate_numpy(audio, 44100)

    np.testing.assert_array_equal(audio, np.full((32, 2), 2.0, dtype=np.float32))
    assert set(stems) == {"Vocals", "Instrumental"}
    for stem in stems.values():
        assert stem.shape == audio.shape
        assert np.isfinite(stem).all()
        assert not np.shares_memory(stem, audio)
    assert numpy_separator.model_instance.primary_source is None
    assert numpy_separator.model_instance.secondary_source is None


@pytest.mark.parametrize("column", [False, True])
def test_numpy_separation_converts_mono_to_stereo(numpy_separator, column):
    audio = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    if column:
        audio = audio[:, None]

    stems = numpy_separator.separate_numpy(audio, 44100)

    for stem in stems.values():
        assert stem.shape == (64, 2)
        np.testing.assert_array_equal(stem[:, 0], stem[:, 1])


def test_numpy_separation_resamples_on_the_frame_axis(numpy_separator):
    audio = np.full((100, 2), 0.25, dtype=np.float32)

    stems = numpy_separator.separate_numpy(audio, 22050)

    assert set(stems) == {"Vocals", "Instrumental"}
    assert all(stem.shape == (200, 2) for stem in stems.values())


def test_numpy_separation_keeps_results_independent_between_calls(numpy_separator):
    first = numpy_separator.separate_numpy(np.full((32, 2), 0.25, dtype=np.float32), 44100)
    saved = {name: stem.copy() for name, stem in first.items()}

    second = numpy_separator.separate_numpy(np.full((32, 2), 0.5, dtype=np.float32), 44100)

    for name, stem in first.items():
        np.testing.assert_array_equal(stem, saved[name])
        assert not np.shares_memory(stem, second[name])
        assert not np.array_equal(stem, second[name])


def test_numpy_separation_honors_single_stem_selection(numpy_separator):
    numpy_separator.model_instance.output_single_stem = "vocals"

    stems = numpy_separator.separate_numpy(np.ones((32, 2), dtype=np.float32), 44100)

    assert set(stems) == {"Vocals"}


def test_numpy_separation_cleans_state_after_inference_failure(numpy_separator):
    model = numpy_separator.model_instance
    model.demix.side_effect = [np.ones((2, 32), dtype=np.float32), RuntimeError("inference failed")]

    with pytest.raises(RuntimeError, match="inference failed"):
        numpy_separator.separate_numpy(np.ones((32, 2), dtype=np.float32), 44100)

    assert model.primary_source is None
    assert model.secondary_source is None
    assert model.audio_file_path is None


@pytest.mark.parametrize(
    "audio,error",
    [
        ("input.wav", TypeError),
        (np.zeros((32, 2), dtype=np.int16), TypeError),
        (np.empty((0, 2), dtype=np.float32), InvalidAudioDataError),
        (np.full((32, 2), np.nan, dtype=np.float32), InvalidAudioDataError),
        (np.zeros((32, 3), dtype=np.float32), InvalidAudioDataError),
    ],
)
def test_numpy_separation_rejects_invalid_audio(numpy_separator, audio, error):
    with pytest.raises(error):
        numpy_separator.separate_numpy(audio, 44100)


@pytest.mark.parametrize("sample_rate", [0, -1, True, 44100.5])
def test_numpy_separation_rejects_invalid_sample_rates(numpy_separator, sample_rate):
    with pytest.raises(ValueError, match="positive integer"):
        numpy_separator.separate_numpy(np.ones((32, 2), dtype=np.float32), sample_rate)


def test_separator_rejects_amplification_above_normalization(tmp_path):
    with pytest.raises(ValueError, match="must not be greater than the normalization_threshold"):
        Separator(output_dir=tmp_path, normalization_threshold=0.5, amplification_threshold=0.8, info_only=True)


def test_numpy_separation_rejects_ensemble_before_any_model_is_loaded(tmp_path):
    separator = Separator(model_file_dir=tmp_path / "models", output_dir=tmp_path / "output", info_only=True)
    separator.load_model(model_filename=["first.ckpt", "second.ckpt"])

    with pytest.raises(NotImplementedError, match="single MDX"):
        separator.separate_numpy(np.ones((32, 2), dtype=np.float32), 44100)
