from unittest.mock import patch

import pytest
import torch

from audio_separator.separator.uvr_lib_v5 import device_utils


@pytest.fixture(autouse=True)
def clear_device_capability_cache():
    device_utils._supports_complex_spectral_ops.cache_clear()
    device_utils._AUTOCAST_SUPPORT_CACHE.clear()
    yield
    device_utils._supports_complex_spectral_ops.cache_clear()
    device_utils._AUTOCAST_SUPPORT_CACHE.clear()


def test_privateuseone_is_not_treated_as_autocast_capable():
    with (
        patch.object(device_utils.torch.amp.autocast_mode, "is_autocast_available", return_value=True) as available,
        patch.object(device_utils.torch, "autocast", side_effect=AssertionError("unsupported")) as autocast,
    ):
        assert device_utils.supports_autocast(torch.device("privateuseone")) is False

    available.assert_not_called()
    autocast.assert_not_called()


def test_autocast_support_requires_a_working_context():
    with (
        patch.object(device_utils.torch.amp.autocast_mode, "is_autocast_available", return_value=True),
        patch.object(device_utils.torch, "autocast", side_effect=AssertionError("unsupported")),
    ):
        assert device_utils.supports_autocast(torch.device("mps")) is False


def test_autocast_support_handles_torch_without_availability_probe():
    with (
        patch.object(device_utils.torch.amp.autocast_mode, "is_autocast_available", None),
        patch.object(device_utils.torch, "autocast") as autocast,
    ):
        assert device_utils.supports_autocast(torch.device("cpu")) is True

    autocast.assert_called_once_with(device_type="cpu", enabled=False)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_standard_devices_are_supported_without_a_runtime_probe(device_type):
    with patch.object(device_utils.torch, "device") as device_constructor:
        assert device_utils._supports_complex_spectral_ops(device_type, -1) is True

    device_constructor.assert_not_called()


def test_directml_short_circuits_without_a_runtime_probe():
    with patch.object(device_utils.torch, "device") as device_constructor:
        assert device_utils._supports_complex_spectral_ops("privateuseone", -1) is False

    device_constructor.assert_not_called()


def test_probe_failure_preserves_cpu_fallback():
    with patch.object(device_utils.torch, "device", side_effect=RuntimeError("unsupported")):
        assert device_utils._supports_complex_spectral_ops("mps", -1) is False


def test_force_cpu_environment_flag_overrides_capability_probe(monkeypatch):
    monkeypatch.setenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", "1")
    with patch.object(device_utils, "_supports_complex_spectral_ops") as probe:
        assert device_utils.should_fallback_to_cpu_for_complex_ops(torch.device("cpu")) is True

    probe.assert_not_called()


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_disabled_force_cpu_environment_values_use_capability_probe(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", raising=False)
    else:
        monkeypatch.setenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", value)

    with patch.object(device_utils, "_supports_complex_spectral_ops", return_value=True) as probe:
        assert device_utils.should_fallback_to_cpu_for_complex_ops(torch.device("mps")) is False

    probe.assert_called_once_with("mps", -1)


def test_fallback_decision_uses_cached_capability_probe(monkeypatch):
    monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", raising=False)
    with patch.object(device_utils, "_supports_complex_spectral_ops", return_value=True) as probe:
        assert device_utils.should_fallback_to_cpu_for_complex_ops(torch.device("mps")) is False

    probe.assert_called_once_with("mps", -1)


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [("cpu", False), ("cuda", False), ("privateuseone", False), ("mps", True)],
)
def test_device_accumulation_is_limited_to_mps(device_type, expected):
    assert device_utils.should_accumulate_on_device(torch.device(device_type), estimated_bytes=1024) is expected


def test_mps_accumulation_falls_back_to_cpu_above_bounded_buffer_size():
    limit = device_utils.MAX_MPS_FULL_TRACK_BUFFER_BYTES

    assert device_utils.should_accumulate_on_device(torch.device("mps"), estimated_bytes=limit) is True
    assert device_utils.should_accumulate_on_device(torch.device("mps"), estimated_bytes=limit + 1) is False


@pytest.mark.parametrize(
    ("device_type", "cac", "complex_fallback", "expected", "probe_called"),
    [
        ("mps", False, False, True, False),
        ("mps", True, False, False, True),
        ("mps", True, True, True, True),
        ("cpu", False, False, False, True),
        ("cuda", False, False, False, True),
    ],
)
def test_demucs_mask_fallback_covers_non_cac_mps_wiener_path(
    device_type,
    cac,
    complex_fallback,
    expected,
    probe_called,
):
    with patch.object(
        device_utils,
        "should_fallback_to_cpu_for_complex_ops",
        return_value=complex_fallback,
    ) as complex_probe:
        assert device_utils.should_fallback_to_cpu_for_demucs_mask(torch.device(device_type), cac) is expected

    assert complex_probe.called is probe_called


def test_probe_rejects_a_device_without_complex_scatter_support():
    cpu_device = torch.device("cpu")
    with (
        patch.object(device_utils.torch, "device", return_value=cpu_device),
        patch.object(device_utils, "_probe_complex_scatter_add", side_effect=RuntimeError("unsupported")),
    ):
        assert device_utils._supports_complex_spectral_ops("mps", -1) is False


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_probe_returns_a_boolean():
    assert isinstance(device_utils._supports_complex_spectral_ops("mps", -1), bool)
