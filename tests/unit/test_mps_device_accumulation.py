from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from audio_separator.separator.architectures.demucs_separator import (
    DemucsSeparator,
    _estimate_demucs_full_track_buffer_bytes,
)
from audio_separator.separator.architectures.mdxc_separator import (
    MDXCSeparator,
    _estimate_mdxc_full_track_buffer_bytes,
    _estimate_roformer_full_track_buffer_bytes,
)
from audio_separator.separator.uvr_lib_v5.device_utils import MAX_MPS_FULL_TRACK_BUFFER_BYTES


def _demucs_separator(device: torch.device) -> DemucsSeparator:
    separator = object.__new__(DemucsSeparator)
    separator.logger = Mock()
    separator.torch_device = device
    separator.demucs_model_instance = Mock()
    separator.demucs_model_instance.sources = ["drums", "bass", "other", "vocals"]
    separator.demucs_model_instance.models = []
    separator.shifts = 0
    separator.segments_enabled = True
    separator.overlap = 0.25
    return separator


def _mix() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 128), dtype=np.float32)


def test_medium_track_buffers_use_mps_but_one_hour_buffers_use_cpu():
    medium_track_samples = round(229.18 * 44100)
    one_hour_samples = 3600 * 44100
    roformer_chunk_size = 485100

    medium_track_estimates = (
        _estimate_roformer_full_track_buffer_bytes(2, 2, medium_track_samples, roformer_chunk_size),
        _estimate_mdxc_full_track_buffer_bytes(2, 2, medium_track_samples + roformer_chunk_size),
        _estimate_demucs_full_track_buffer_bytes(2, medium_track_samples, 4, shifts=2, num_bag_models=1),
    )
    one_hour_estimates = (
        _estimate_roformer_full_track_buffer_bytes(2, 2, one_hour_samples, roformer_chunk_size),
        _estimate_mdxc_full_track_buffer_bytes(2, 2, one_hour_samples + roformer_chunk_size),
        _estimate_demucs_full_track_buffer_bytes(2, one_hour_samples, 4, shifts=2, num_bag_models=1),
    )

    assert all(estimated <= MAX_MPS_FULL_TRACK_BUFFER_BYTES for estimated in medium_track_estimates)
    assert all(estimated > MAX_MPS_FULL_TRACK_BUFFER_BYTES for estimated in one_hour_estimates)


class _FakeMDXCModel:
    num_target_instruments = 2

    def __init__(self, expected_device: torch.device):
        self.expected_device = expected_device

    def __call__(self, batch):
        assert batch.device.type == self.expected_device.type
        return batch.unsqueeze(1).repeat(1, 2, 1, 1)


class _FakeRoformerModel(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device=device))

    def forward(self, batch):
        assert batch.device == self.anchor.device
        return batch.unsqueeze(1).repeat(1, 2, 1, 1)


def _mdxc_separator(device: torch.device) -> MDXCSeparator:
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.torch_device = device
    separator.model_run = _FakeMDXCModel(device)
    separator.model_data_cfgdict = SimpleNamespace(
        training=SimpleNamespace(instruments=["first", "second"], target_instrument=None),
        inference=SimpleNamespace(dim_t=5),
        audio=SimpleNamespace(hop_length=2),
    )
    separator.pitch_shift = 0
    separator.is_roformer = False
    separator.segment_size = 5
    separator.overlap = 2
    separator.batch_size = 1
    separator.is_primary_stem_main_target = False
    return separator


def _roformer_separator(device: torch.device) -> MDXCSeparator:
    separator = _mdxc_separator(device)
    separator.model_run = _FakeRoformerModel(device)
    separator.model_data_cfgdict.model = SimpleNamespace(stft_hop_length=2)
    separator.model_data_cfgdict.audio.sample_rate = 1
    separator.is_roformer = True
    separator.overlap = 8
    return separator


def test_demucs_keeps_full_track_input_on_cpu_for_cpu_inference():
    separator = _demucs_separator(torch.device("cpu"))

    def fake_apply_model(*, mix, **kwargs):
        assert mix.device.type == "cpu"
        return torch.zeros(1, 4, 2, mix.shape[-1], device=mix.device)

    with patch("audio_separator.separator.architectures.demucs_separator.apply_model", side_effect=fake_apply_model):
        result = separator.demix_demucs(_mix())

    assert result.shape == (4, 2, 128)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_demucs_keeps_full_track_input_on_mps():
    separator = _demucs_separator(torch.device("mps"))

    def fake_apply_model(*, mix, **kwargs):
        assert mix.device.type == "mps"
        return torch.zeros(1, 4, 2, mix.shape[-1], device=mix.device)

    with patch("audio_separator.separator.architectures.demucs_separator.apply_model", side_effect=fake_apply_model):
        result = separator.demix_demucs(_mix())

    assert result.shape == (4, 2, 128)


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_mdxc_chunk_buffers_share_the_selected_accumulation_device(device_type):
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    device = torch.device(device_type)
    separator = _mdxc_separator(device)
    allocated_devices = []
    torch_zeros = torch.zeros

    def tracked_zeros(*args, **kwargs):
        tensor = torch_zeros(*args, **kwargs)
        allocated_devices.append(tensor.device.type)
        return tensor

    with patch("audio_separator.separator.architectures.mdxc_separator.torch.zeros", side_effect=tracked_zeros):
        result = separator.demix(_mix(), override_model_segment_size=True)

    assert set(result) == {"first", "second"}
    assert all(stem.shape == (2, 128) for stem in result.values())
    assert allocated_devices
    assert set(allocated_devices) == {device_type}


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_roformer_overlap_add_buffers_use_the_selected_accumulation_device(device_type):
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    separator = _roformer_separator(torch.device(device_type))
    window_devices = []
    zero_devices = []
    torch_tensor = torch.tensor
    torch_zeros = torch.zeros

    def tracked_tensor(*args, **kwargs):
        tensor = torch_tensor(*args, **kwargs)
        if kwargs.get("device") is not None:
            window_devices.append(tensor.device.type)
        return tensor

    def tracked_zeros(*args, **kwargs):
        tensor = torch_zeros(*args, **kwargs)
        zero_devices.append(tensor.device.type)
        return tensor

    with (
        patch("audio_separator.separator.architectures.mdxc_separator.torch.tensor", side_effect=tracked_tensor),
        patch("audio_separator.separator.architectures.mdxc_separator.torch.zeros", side_effect=tracked_zeros),
    ):
        result = separator.demix(_mix(), override_model_segment_size=True)

    assert set(result) == {"first", "second"}
    assert window_devices == [device_type]
    assert zero_devices == [device_type, device_type]
