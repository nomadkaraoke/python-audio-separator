from unittest.mock import Mock, patch

import pytest
import torch

from audio_separator.separator.execution_policy import AUTOCAST, FP32, NATIVE_FP16, resolve_execution_policy
from audio_separator.separator.separator import Separator


def _resolve(*, device="mps", requested_device=None, model="mel_band_roformer", autocast=False, native=False, compile=False, pytorch=True):
    logger = Mock()
    policy = resolve_execution_policy(
        device=torch.device(device),
        requested_device=torch.device(requested_device) if requested_device else None,
        model_family=model,
        use_autocast=autocast,
        use_native_fp16=native,
        use_torch_compile=compile,
        uses_pytorch_inference=pytorch,
        logger=logger,
    )
    return policy, logger


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("model", ["mel_band_roformer", "bs_roformer"])
def test_verified_roformer_native_fp16_is_enabled_on_accelerators(device, model):
    policy, logger = _resolve(device=device, model=model, native=True)

    assert policy.precision == NATIVE_FP16
    assert policy.use_torch_compile is False
    logger.warning.assert_not_called()


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("model", ["mel_band_roformer", "bs_roformer"])
@pytest.mark.parametrize(
    ("autocast", "native", "expected_precision"),
    [
        (False, False, FP32),
        (True, False, AUTOCAST),
        (False, True, NATIVE_FP16),
    ],
)
def test_verified_roformer_precision_modes_can_be_combined_with_compile(
    device,
    model,
    autocast,
    native,
    expected_precision,
):
    with patch("audio_separator.separator.execution_policy.supports_autocast", return_value=True):
        policy, logger = _resolve(
            device=device,
            model=model,
            autocast=autocast,
            native=native,
            compile=True,
        )

    assert policy.precision == expected_precision
    assert policy.use_torch_compile is True
    logger.warning.assert_not_called()


@pytest.mark.parametrize("model", ["mel_band_roformer", "bs_roformer"])
@pytest.mark.parametrize(
    ("autocast", "expected_precision"),
    [
        (False, FP32),
        (True, AUTOCAST),
    ],
)
def test_verified_cpu_roformer_modes_can_be_combined_with_compile(model, autocast, expected_precision):
    with patch("audio_separator.separator.execution_policy.supports_autocast", return_value=True):
        policy, logger = _resolve(
            device="cpu",
            model=model,
            autocast=autocast,
            compile=True,
        )

    assert policy.precision == expected_precision
    assert policy.use_torch_compile is True
    logger.warning.assert_not_called()


def test_unsupported_cpu_native_fp16_falls_back_to_verified_fp32_compile():
    policy, logger = _resolve(device="cpu", native=True, compile=True)

    assert policy.precision == FP32
    assert policy.use_torch_compile is True
    logger.warning.assert_called_once()


@pytest.mark.parametrize("model", ["vr", "demucs"])
def test_unverified_models_reject_native_fp16_and_compile(model):
    policy, logger = _resolve(model=model, native=True, compile=True)

    assert policy.precision == FP32
    assert policy.use_torch_compile is False
    assert logger.warning.call_count == 2


@pytest.mark.parametrize("model", ["vr", "demucs"])
def test_unverified_models_keep_existing_autocast_but_reject_compile(model):
    with patch("audio_separator.separator.execution_policy.supports_autocast", return_value=True):
        policy, logger = _resolve(model=model, autocast=True, compile=True)

    assert policy.precision == AUTOCAST
    assert policy.use_torch_compile is False
    logger.warning.assert_called_once()


def test_autocast_and_compile_are_orthogonal_requests():
    with patch("audio_separator.separator.execution_policy.supports_autocast", return_value=True):
        policy, logger = _resolve(autocast=True, compile=True)

    assert policy.precision == AUTOCAST
    assert policy.use_torch_compile is True
    logger.warning.assert_not_called()


def test_directml_never_enables_autocast():
    with patch("audio_separator.separator.execution_policy.supports_autocast") as available:
        policy, logger = _resolve(device="privateuseone", autocast=True)

    assert policy.precision == FP32
    assert policy.use_torch_compile is False
    available.assert_not_called()
    logger.warning.assert_called_once()


def test_onnx_runtime_model_does_not_report_autocast_as_effective():
    with patch("audio_separator.separator.execution_policy.supports_autocast") as available:
        policy, logger = _resolve(device="mps", model="mdx", autocast=True, pytorch=False)

    assert policy.precision == FP32
    assert policy.use_torch_compile is False
    available.assert_not_called()
    logger.warning.assert_called_once()


def test_directml_cpu_fallback_still_uses_float32_eager():
    with patch("audio_separator.separator.execution_policy.supports_autocast") as available:
        policy, logger = _resolve(
            device="cpu",
            requested_device="privateuseone",
            model="bs_roformer",
            autocast=True,
            compile=True,
        )

    assert policy.precision == FP32
    assert policy.use_torch_compile is False
    available.assert_not_called()
    assert logger.warning.call_count == 2


def test_directml_cpu_fallback_rejects_native_fp16_and_compile():
    policy, logger = _resolve(
        device="cpu",
        requested_device="privateuseone",
        model="bs_roformer",
        native=True,
        compile=True,
    )

    assert policy.precision == FP32
    assert policy.use_torch_compile is False
    assert logger.warning.call_count == 2


def test_resolver_rejects_conflicting_precision_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve(autocast=True, native=True)


def test_constructor_rejects_conflicting_precision_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Separator(info_only=True, use_autocast=True, use_native_fp16=True)


def test_effective_properties_follow_loaded_model_state():
    separator = object.__new__(Separator)
    separator.model_instance = None

    assert separator.effective_precision == FP32
    assert separator.effective_torch_compile is False

    separator.model_instance = Mock(effective_precision=NATIVE_FP16, effective_torch_compile=True)

    assert separator.effective_precision == NATIVE_FP16
    assert separator.effective_torch_compile is True
