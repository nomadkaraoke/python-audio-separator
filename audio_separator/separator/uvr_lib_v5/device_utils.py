"""Device capability helpers for hardware-specific inference paths."""

import os
from contextlib import nullcontext
from functools import lru_cache

import torch


_AUTOCAST_SUPPORT_CACHE = {}

# Full-track buffers grow linearly with audio duration. Keep short and medium
# inputs on MPS, but leave oversized accumulators on CPU so they cannot consume
# most of the MPS working-set budget before model activations are allocated.
MAX_MPS_FULL_TRACK_BUFFER_BYTES = 1024**3


def _supports_autocast(device_type: str) -> bool:
    """Return whether PyTorch can safely enter autocast for a device type."""
    if device_type in _AUTOCAST_SUPPORT_CACHE:
        return _AUTOCAST_SUPPORT_CACHE[device_type]

    # PyTorch reports privateuseone as autocast-capable even though
    # torch-directml does not register the AMP hooks required by the context.
    if device_type == "privateuseone":
        _AUTOCAST_SUPPORT_CACHE[device_type] = False
        return False

    try:
        is_available = getattr(torch.amp.autocast_mode, "is_autocast_available", None)
        if is_available is not None and not is_available(device_type):
            _AUTOCAST_SUPPORT_CACHE[device_type] = False
            return False
        with torch.autocast(device_type=device_type, enabled=False):
            pass
        supported = True
    except (AssertionError, RuntimeError, TypeError, ValueError):
        supported = False

    _AUTOCAST_SUPPORT_CACHE[device_type] = supported
    return supported


def supports_autocast(device: torch.device) -> bool:
    """Return whether a device can use PyTorch's generic autocast context."""
    return _supports_autocast(device.type)


def autocast_disabled(device: torch.device):
    """Disable autocast when supported, otherwise return a no-op context."""
    if not supports_autocast(device):
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=False)


def _probe_complex_scatter_add(spectrum: torch.Tensor) -> None:
    """Exercise the complex scatter operation used by MelBand RoFormer."""
    source = spectrum[:, :2, :2]
    indices = torch.zeros(source.shape, dtype=torch.long, device=source.device)
    torch.zeros_like(source).scatter_add_(1, indices, source)


@lru_cache(maxsize=32)
def _supports_complex_spectral_ops(device_type: str, device_index: int) -> bool:
    """Return whether a device can execute the complex operations used by the models."""
    if device_type in {"cpu", "cuda"}:
        return True

    # DirectML cannot represent complex tensors. Avoid probing unsupported
    # operations on its out-of-tree backend slot.
    if device_type == "privateuseone":
        return False

    try:
        device = torch.device(f"{device_type}:{device_index}") if device_index >= 0 else torch.device(device_type)
        sample_length = 1024
        n_fft = 256
        hop_length = 64
        sample = torch.randn(1, sample_length, device=device)
        window = torch.hann_window(n_fft, device=device)
        spectrum = torch.stft(sample, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=True)
        spectrum = torch.view_as_complex(torch.view_as_real(spectrum).contiguous()) * torch.ones_like(spectrum)
        _probe_complex_scatter_add(spectrum)
        torch.istft(spectrum, n_fft=n_fft, hop_length=hop_length, window=window, center=True, length=sample_length)
        return True
    except Exception:
        return False


def should_fallback_to_cpu_for_complex_ops(device: torch.device) -> bool:
    """Return whether complex spectral operations should use the legacy CPU path."""
    if os.environ.get("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX") == "1":
        return True

    device_index = -1 if device.index is None else int(device.index)
    return not _supports_complex_spectral_ops(device.type, device_index)


def should_fallback_to_cpu_for_demucs_mask(device: torch.device, cac: bool) -> bool:
    """Keep non-CaC Demucs Wiener masking on CPU because the spectral probe does not cover it."""
    return (device.type == "mps" and not cac) or should_fallback_to_cpu_for_complex_ops(device)


def should_accumulate_on_device(device: torch.device, estimated_bytes: int) -> bool:
    """Return whether duration-scaled buffers fit the bounded MPS fast path."""
    return device.type == "mps" and estimated_bytes <= MAX_MPS_FULL_TRACK_BUFFER_BYTES
