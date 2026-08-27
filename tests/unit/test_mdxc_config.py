from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.common_separator import CommonSeparator


def _make_separator(arch_config, inference_config=None):
    separator = MDXCSeparator.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_data = {
        "inference": inference_config or {},
        "training": {"target_instrument": "Vocals"},
    }
    separator.torch_device = torch.device("cpu")
    separator.torch_device_cpu = torch.device("cpu")

    def load_model():
        separator.model_data_cfgdict = ConfigDict(separator.model_data)

    with patch.object(CommonSeparator, "__init__", return_value=None), patch.object(MDXCSeparator, "load_model", side_effect=load_model):
        MDXCSeparator.__init__(separator, {}, arch_config)

    return separator


def test_mdxc_inference_defaults_come_from_model_config():
    separator = _make_separator({"overlap": None, "batch_size": None}, {"num_overlap": 2, "batch_size": 4})

    assert separator.overlap == 2
    assert separator.batch_size == 4


def test_mdxc_explicit_inference_options_override_model_config():
    separator = _make_separator({"overlap": 8, "batch_size": 1}, {"num_overlap": 2, "batch_size": 4})

    assert separator.overlap == 8
    assert separator.batch_size == 1


def test_mdxc_inference_defaults_fall_back_for_older_configs():
    separator = _make_separator({"overlap": None, "batch_size": None})

    assert separator.overlap == 8
    assert separator.batch_size == 1


def test_mdxc_null_inference_values_use_fallbacks():
    separator = _make_separator({"overlap": None, "batch_size": None}, {"num_overlap": None, "batch_size": None})

    assert separator.overlap == 8
    assert separator.batch_size == 1


@pytest.mark.parametrize(
    ("arch_config", "inference_config", "message"),
    [
        ({"overlap": 0, "batch_size": 1}, {"num_overlap": 2}, "MDXC overlap"),
        ({"overlap": -1, "batch_size": 1}, {"num_overlap": 2}, "MDXC overlap"),
        ({"overlap": None, "batch_size": 1}, {"num_overlap": 0}, "MDXC overlap"),
        ({"overlap": None, "batch_size": 1}, {"num_overlap": -1}, "MDXC overlap"),
        ({"overlap": 2, "batch_size": 0}, {"batch_size": 1}, "MDXC batch size"),
        ({"overlap": 2, "batch_size": -1}, {"batch_size": 1}, "MDXC batch size"),
        ({"overlap": 2, "batch_size": None}, {"batch_size": 0}, "MDXC batch size"),
        ({"overlap": 2, "batch_size": None}, {"batch_size": -1}, "MDXC batch size"),
    ],
)
def test_mdxc_rejects_non_positive_inference_values(arch_config, inference_config, message):
    with pytest.raises(ValueError, match=message):
        _make_separator(arch_config, inference_config)


def test_mdxc_non_roformer_rejects_overlap_larger_than_chunk_size():
    separator = MDXCSeparator.__new__(MDXCSeparator)
    separator.pitch_shift = 0
    separator.is_roformer = False
    separator.override_model_segment_size = False
    separator.model_data_cfgdict = ConfigDict(
        {
            "inference": {"dim_t": 5},
            "audio": {"hop_length": 2},
        }
    )
    separator.overlap = 9
    separator.model_run = Mock(num_target_instruments=1)
    separator.logger = Mock()

    with pytest.raises(ValueError, match=r"MDXC overlap \(9\) must not exceed chunk size \(8\)"):
        separator.demix(np.ones((2, 16), dtype=np.float32))
