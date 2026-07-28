from unittest.mock import patch

import pytest
import torch
from rotary_embedding_torch import RotaryEmbedding
from rotary_embedding_torch.rotary_embedding_torch import rotate_half

from audio_separator.separator.uvr_lib_v5 import device_utils
from audio_separator.separator.uvr_lib_v5.roformer import bs_roformer as bs_mod
from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mel_mod
from audio_separator.separator.uvr_lib_v5.roformer import rotary as rotary_mod


def _float32_reference(rotary: RotaryEmbedding, tensor: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(tensor.shape[-2], device=tensor.device, dtype=torch.float32)
    angles = torch.einsum("n, f -> n f", positions, rotary.freqs.float())
    angles = torch.repeat_interleave(angles, 2, dim=-1)
    rotated = tensor.float() * angles.cos() + rotate_half(tensor.float()) * angles.sin()
    return rotated.to(tensor.dtype)


def _seed_low_precision_cache(rotary: RotaryEmbedding, *, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
    rotary.cached_freqs = torch.zeros(seq_len, rotary.freqs.numel() * 2, device=device, dtype=dtype)


def test_mel_and_bs_use_the_shared_rotary_helper():
    assert mel_mod._rotate_queries_or_keys is rotary_mod.rotate_queries_or_keys
    assert bs_mod._rotate_queries_or_keys is rotary_mod.rotate_queries_or_keys


def test_cpu_autocast_replaces_low_precision_cache_and_preserves_input_dtype():
    torch.manual_seed(0)
    rotary = RotaryEmbedding(dim=64)
    tensor = torch.randn(2, 4, 1101, 64, dtype=torch.bfloat16)
    _seed_low_precision_cache(rotary, seq_len=1101, device=tensor.device, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = rotary_mod.rotate_queries_or_keys(rotary, tensor)

    expected = _float32_reference(rotary, tensor)
    assert actual.dtype == tensor.dtype
    assert rotary.cached_freqs.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cached_freqs = rotary.cached_freqs
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        repeated = rotary_mod.rotate_queries_or_keys(rotary, tensor)

    assert rotary.cached_freqs is cached_freqs
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)


def test_compiled_rotation_bypasses_mutable_frequency_cache():
    torch.manual_seed(0)
    rotary = RotaryEmbedding(dim=8)
    tensor = torch.randn(1, 2, 11, 8)
    cached_freqs = torch.zeros(11, 8)
    rotary.cached_freqs = cached_freqs

    with patch.object(torch.compiler, "is_compiling", return_value=True):
        actual = rotary_mod.rotate_queries_or_keys(rotary, tensor)

    assert rotary.cached_freqs is cached_freqs
    torch.testing.assert_close(actual, _float32_reference(rotary, tensor))


def test_rotation_does_not_open_autocast_for_an_unsupported_backend():
    torch.manual_seed(0)
    rotary = RotaryEmbedding(dim=8)
    tensor = torch.randn(1, 4, 8)

    with (
        patch.object(device_utils, "supports_autocast", return_value=False),
        patch.object(device_utils.torch, "autocast", side_effect=AssertionError("autocast must not be opened")),
    ):
        actual = rotary_mod.rotate_queries_or_keys(rotary, tensor)

    assert actual.shape == tensor.shape


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_autocast_uses_float32_angles_and_replaces_low_precision_cache():
    torch.manual_seed(0)
    rotary = RotaryEmbedding(dim=64).to("mps")
    tensor = torch.randn(2, 4, 1101, 64, device="mps", dtype=torch.float16)
    _seed_low_precision_cache(rotary, seq_len=1101, device=tensor.device, dtype=torch.float16)

    with torch.autocast(device_type="mps", dtype=torch.float16):
        actual = rotary_mod.rotate_queries_or_keys(rotary, tensor)

    expected = _float32_reference(rotary, tensor)
    assert actual.dtype == tensor.dtype
    assert rotary.cached_freqs.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
