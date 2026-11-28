import logging
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from audio_separator.separator import Separator
from audio_separator.separator.architectures import demucs_separator
from audio_separator.separator.architectures.demucs_separator import DEMUCS_4_SOURCE_MAPPER, DemucsSeparator
from audio_separator.separator.execution_policy import FP32

STEMS_BY_SOURCE_COUNT = {
    2: {"Instrumental", "Vocals"},
    4: {"Bass", "Drums", "Other", "Vocals"},
    6: {"Bass", "Drums", "Other", "Vocals", "Guitar", "Piano"},
}


def fake_demix(num_sources):
    """Return a demix stand-in whose source i is the mix scaled by (i + 1) / 10."""
    return lambda mix: np.stack([mix * ((index + 1) / 10) for index in range(num_sources)])


@pytest.fixture
def demucs_numpy_separator(tmp_path, monkeypatch):
    for method in ("load", "get_duration"):
        monkeypatch.setattr(
            f"audio_separator.separator.common_separator.librosa.{method}",
            Mock(side_effect=AssertionError("Unexpected audio file read")),
        )
    # Only model loading and inference are mocked; stem mapping and post-processing run for real.
    monkeypatch.setattr(demucs_separator, "HDemucs", Mock())
    monkeypatch.setattr(demucs_separator, "get_demucs_model", Mock())
    monkeypatch.setattr(demucs_separator, "demucs_segments", lambda segment_size, model: model)

    separator = Separator(model_file_dir=tmp_path / "models", output_dir=tmp_path / "output", info_only=True)
    model = DemucsSeparator.__new__(DemucsSeparator)
    model.logger = logging.getLogger(__name__)
    model.torch_device = torch.device("cpu")
    model.effective_precision = FP32
    model.sample_rate = 44100
    model.normalization_threshold = 0.9
    model.amplification_threshold = 0.0
    model.output_single_stem = None
    model.input_bit_depth = None
    model.input_subtype = None
    model.model_name = "test"
    model.model_path = str(tmp_path / "test.yaml")
    model.segment_size = "Default"
    model.demucs_source_map = DEMUCS_4_SOURCE_MAPPER
    model.demix_demucs = Mock(side_effect=fake_demix(4))
    model.clear_gpu_cache = Mock()
    model.write_audio = Mock(side_effect=AssertionError("Unexpected audio file write"))
    separator.torch_device = model.torch_device
    separator.model_instance = model
    separator.model_filename = "test.yaml"
    separator.model_filenames = ["test.yaml"]
    return separator


@pytest.mark.parametrize("num_sources", [2, 4, 6])
def test_demucs_numpy_maps_every_source_to_its_stem(demucs_numpy_separator, num_sources):
    model = demucs_numpy_separator.model_instance
    model.demix_demucs.side_effect = fake_demix(num_sources)
    audio = np.full((32, 2), 0.5, dtype=np.float32)
    audio.flags.writeable = False

    stems = demucs_numpy_separator.separate_numpy(audio, 44100)

    np.testing.assert_array_equal(audio, np.full((32, 2), 0.5, dtype=np.float32))
    assert set(stems) == STEMS_BY_SOURCE_COUNT[num_sources]
    for stem_name, stem in stems.items():
        index = model.demucs_source_map[stem_name]
        np.testing.assert_allclose(stem, audio * ((index + 1) / 10), rtol=1e-6)
        assert not np.shares_memory(stem, audio)


def test_demucs_numpy_honors_single_stem_selection(demucs_numpy_separator):
    demucs_numpy_separator.model_instance.output_single_stem = "vocals"

    stems = demucs_numpy_separator.separate_numpy(np.full((32, 2), 0.5, dtype=np.float32), 44100)

    assert set(stems) == {"Vocals"}


def test_demucs_numpy_converts_mono_to_stereo(demucs_numpy_separator):
    stems = demucs_numpy_separator.separate_numpy(np.linspace(-0.5, 0.5, 64, dtype=np.float32), 44100)

    for stem in stems.values():
        assert stem.shape == (64, 2)
        np.testing.assert_array_equal(stem[:, 0], stem[:, 1])


def test_demucs_file_mode_still_loads_and_writes_audio_files(demucs_numpy_separator, monkeypatch, tmp_path):
    model = demucs_numpy_separator.model_instance
    model.output_format = "FLAC"
    model.write_audio = Mock()
    load = Mock(return_value=(np.full((2, 32), 0.5, dtype=np.float32), 44100))
    monkeypatch.setattr("audio_separator.separator.common_separator.librosa.load", load)
    input_path = str(tmp_path / "song.flac")

    output_files = model.separate(input_path)

    assert output_files == [f"song_({name})_test.flac" for name in ("Bass", "Drums", "Other", "Vocals")]
    assert load.call_args.args[0] == input_path
    assert model.write_audio.call_count == 4
