"""Architecture-level contracts for propagating final audio writer failures."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.exceptions import AudioExportError


def architecture_double(separator_class):
    separator = Mock(spec=separator_class)
    separator.logger = logging.getLogger(__name__)
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.output_single_stem = None
    return separator


def test_mdx_does_not_return_planned_path_when_final_writer_fails():
    separator = architecture_double(MDXSeparator)
    separator.prepare_mix.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.demix.side_effect = [np.zeros((2, 32), dtype=np.float32), np.zeros((2, 32), dtype=np.float32)]
    separator.primary_source = None
    separator.secondary_source = None
    separator.primary_stem_name = "Vocals"
    separator.secondary_stem_name = "Instrumental"
    separator.compensate = 1.0
    separator.invert_using_spec = False
    separator.get_stem_output_path.side_effect = ["input_(Instrumental).wav", "input_(Vocals).wav"]
    writer_error = AudioExportError("writer failed", path="input_(Instrumental).wav", backend="pydub")
    separator.final_process.side_effect = writer_error

    with pytest.raises(AudioExportError) as raised:
        MDXSeparator.separate(separator, "input.wav")

    assert raised.value is writer_error


def test_mdxc_multistem_does_not_return_planned_path_when_final_writer_fails():
    separator = architecture_double(MDXCSeparator)
    separator.prepare_mix.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.sample_rate = 44100
    separator.override_model_segment_size = True
    separator.process_all_stems = True
    separator.primary_source = None
    separator.secondary_source = None
    separator.model_data_cfgdict = SimpleNamespace(
        training=SimpleNamespace(target_instrument=None, instruments=["Vocals", "Drums", "Bass"])
    )
    separator.demix.return_value = {
        "Vocals": np.zeros((2, 32), dtype=np.float32),
        "Drums": np.zeros((2, 32), dtype=np.float32),
        "Bass": np.zeros((2, 32), dtype=np.float32),
    }
    separator.get_stem_output_path.return_value = "input_(Vocals).wav"
    writer_error = AudioExportError("writer failed", path="input_(Vocals).wav", backend="soundfile")
    separator.final_process.side_effect = writer_error

    with pytest.raises(AudioExportError) as raised:
        MDXCSeparator.separate(separator, "input.wav")

    assert raised.value is writer_error
