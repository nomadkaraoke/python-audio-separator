import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from audio_separator.separator.architectures.demucs_separator import DemucsSeparator


def test_demucs_model_is_released_after_inference_failure():
    separator = object.__new__(DemucsSeparator)
    separator.logger = logging.getLogger(__name__)
    separator.model_path = "/tmp/htdemucs.yaml"
    separator.segment_size = "Default"
    separator.torch_device = torch.device("cpu")
    separator.prepare_mix = MagicMock(return_value=np.zeros((2, 16), dtype=np.float32))
    separator.demix_demucs = MagicMock(side_effect=RuntimeError("demix failed"))
    separator.clear_gpu_cache = MagicMock(side_effect=RuntimeError("cleanup failed"))

    model = MagicMock(spec=torch.nn.Module)
    model.sources = ["drums", "bass", "other", "vocals"]

    with (
        patch("audio_separator.separator.architectures.demucs_separator.HDemucs"),
        patch("audio_separator.separator.architectures.demucs_separator.get_demucs_model", return_value=model),
        patch("audio_separator.separator.architectures.demucs_separator.demucs_segments", return_value=model),
    ):
        with pytest.raises(RuntimeError, match="demix failed"):
            separator.separate("input.wav")

    assert not hasattr(separator, "demucs_model_instance")
    separator.clear_gpu_cache.assert_called_once_with()
