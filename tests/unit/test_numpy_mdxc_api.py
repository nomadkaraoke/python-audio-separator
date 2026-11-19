import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from audio_separator.separator import Separator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.execution_policy import FP32


@pytest.fixture
def mdxc_numpy_separator(tmp_path, monkeypatch):
    separator = Separator(model_file_dir=tmp_path / "models", output_dir=tmp_path / "output", info_only=True)
    model = MDXCSeparator.__new__(MDXCSeparator)
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
    model.process_all_stems = True
    model.override_model_segment_size = False
    model.model_data_cfgdict = SimpleNamespace(
        training=SimpleNamespace(target_instrument="Vocals", instruments=["Vocals", "Instrumental"])
    )
    model.demix = Mock(side_effect=lambda mix, **kwargs: {"Vocals": mix * 0.25, "Instrumental": mix * 0.75})
    model.clear_gpu_cache = Mock()
    model.write_audio = Mock(side_effect=AssertionError("Unexpected audio file write"))
    for method in ("load", "get_duration"):
        monkeypatch.setattr(
            f"audio_separator.separator.common_separator.librosa.{method}",
            Mock(side_effect=AssertionError("Unexpected audio file read")),
        )
    separator.torch_device = model.torch_device
    separator.model_instance = model
    separator.model_filename = "test.ckpt"
    separator.model_filenames = ["test.ckpt"]
    return separator


def test_mdxc_numpy_returns_stereo_arrays_without_changing_input(mdxc_numpy_separator):
    audio = np.full((32, 2), 0.5, dtype=np.float32)
    audio.flags.writeable = False

    stems = mdxc_numpy_separator.separate_numpy(audio, 44100)

    np.testing.assert_array_equal(audio, np.full((32, 2), 0.5, dtype=np.float32))
    np.testing.assert_allclose(stems["Vocals"], audio * 0.25)
    np.testing.assert_allclose(stems["Instrumental"], audio * 0.75)
    assert all(stem.shape == audio.shape for stem in stems.values())


@pytest.mark.parametrize("selected_stem", [None, "bass"])
def test_mdxc_numpy_returns_selected_multistem_outputs(mdxc_numpy_separator, selected_stem):
    model = mdxc_numpy_separator.model_instance
    model.model_data_cfgdict.training = SimpleNamespace(target_instrument=None, instruments=["Vocals", "Drums", "Bass"])
    model.output_single_stem = selected_stem
    model.demix.side_effect = lambda mix, **kwargs: {
        "Vocals": mix * 0.25,
        "Drums": mix * 0.5,
        "Bass": mix * 0.75,
    }

    stems = mdxc_numpy_separator.separate_numpy(np.ones((32, 2), dtype=np.float32), 44100)

    assert set(stems) == ({"Bass"} if selected_stem else {"Vocals", "Drums", "Bass"})
    for stem in stems.values():
        assert stem.shape == (32, 2)
        assert np.isfinite(stem).all()


def test_mdxc_numpy_returns_single_array_source(mdxc_numpy_separator):
    mdxc_numpy_separator.model_instance.demix.side_effect = lambda mix, **kwargs: mix * 0.25

    stems = mdxc_numpy_separator.separate_numpy(np.ones((32, 2), dtype=np.float32), 44100)

    assert set(stems) == {"Vocals"}
    assert stems["Vocals"].shape == (32, 2)


def test_mdxc_numpy_results_remain_independent_between_calls(mdxc_numpy_separator):
    first = mdxc_numpy_separator.separate_numpy(np.full((32, 2), 0.25, dtype=np.float32), 44100)
    saved = {name: stem.copy() for name, stem in first.items()}

    second = mdxc_numpy_separator.separate_numpy(np.full((32, 2), 0.5, dtype=np.float32), 44100)

    for name, stem in first.items():
        np.testing.assert_array_equal(stem, saved[name])
        assert not np.shares_memory(stem, second[name])
        assert not np.array_equal(stem, second[name])
    assert mdxc_numpy_separator.model_instance.override_model_segment_size is False
