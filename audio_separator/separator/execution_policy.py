"""Resolve requested inference options against verified execution capabilities."""

from dataclasses import dataclass

from packaging import version
import torch

from audio_separator.separator.uvr_lib_v5.device_utils import supports_autocast

FP32 = "fp32"
AUTOCAST = "autocast"
NATIVE_FP16 = "native_fp16"
MIN_TORCH_COMPILE_VERSION = version.parse("2.6")

# Keep these tables intentionally conservative. A cell should only be added after
# its correctness, numerical quality, and fallback behavior have been verified.
# Speed remains workload-dependent, especially when compilation has a cold start.
NATIVE_FP16_CAPABILITIES = frozenset(
    {
        ("mps", "mel_band_roformer"),
        ("mps", "bs_roformer"),
        ("cuda", "mel_band_roformer"),
        ("cuda", "bs_roformer"),
    }
)

TORCH_COMPILE_CAPABILITIES = frozenset(
    {
        (device, model_family, precision)
        for device in ("mps", "cuda")
        for model_family in ("mel_band_roformer", "bs_roformer")
        for precision in (FP32, AUTOCAST, NATIVE_FP16)
    }
    | {
        ("cpu", model_family, precision)
        for model_family in ("mel_band_roformer", "bs_roformer")
        for precision in (FP32, AUTOCAST)
    }
)


@dataclass(frozen=True)
class ExecutionPolicy:
    """The effective execution choices for one loaded model."""

    precision: str = FP32
    use_torch_compile: bool = False


def _regional_compile_runtime_supported() -> bool:
    """Return whether Dynamo can trace the SDPA context used by RoFormer."""
    torch_version = version.parse(torch.__version__.split("+")[0])
    return hasattr(torch, "compile") and torch_version >= MIN_TORCH_COMPILE_VERSION


def resolve_execution_policy(
    *,
    device,
    requested_device=None,
    model_family: str,
    use_autocast: bool,
    use_native_fp16: bool,
    use_torch_compile: bool,
    logger,
    uses_pytorch_inference: bool = True,
) -> ExecutionPolicy:
    """Resolve requested options, warning when an unverified path is skipped."""
    if use_autocast and use_native_fp16:
        raise ValueError("Autocast and native float16 are mutually exclusive precision modes.")

    device_type = getattr(device, "type", str(device))
    requested_device_type = getattr(requested_device, "type", str(requested_device)) if requested_device is not None else device_type
    capability_device_type = requested_device_type if requested_device_type == "privateuseone" else device_type
    normalized_family = (model_family or "unknown").lower()
    precision = FP32

    if use_native_fp16:
        capability = (capability_device_type, normalized_family)
        if capability in NATIVE_FP16_CAPABILITIES:
            precision = NATIVE_FP16
        else:
            logger.warning(
                "Native float16 is not supported for device=%s, model=%s; continuing with float32 inference.",
                device_type,
                normalized_family,
            )
    elif use_autocast:
        if not uses_pytorch_inference:
            logger.warning("Autocast only applies to PyTorch inference; continuing with the model's native precision.")
        # torch-directml exposes its device as privateuseone. PyTorch's generic
        # autocast context does not support that backend, so never enter it.
        elif requested_device_type == "privateuseone":
            logger.warning("Autocast is not supported on DirectML; continuing with float32 inference.")
        elif supports_autocast(device):
            precision = AUTOCAST
        else:
            logger.warning("Autocast is not available for device=%s; continuing with float32 inference.", device_type)

    compile_enabled = False
    if use_torch_compile:
        capability = (capability_device_type, normalized_family, precision)
        if capability not in TORCH_COMPILE_CAPABILITIES:
            logger.warning(
                "Regional torch.compile is not supported for device=%s, model=%s, precision=%s; continuing with eager inference.",
                device_type,
                normalized_family,
                precision,
            )
        elif not _regional_compile_runtime_supported():
            logger.warning(
                "Regional torch.compile requires PyTorch 2.6 or newer; found %s. Continuing with eager inference.",
                torch.__version__,
            )
        else:
            compile_enabled = True

    return ExecutionPolicy(precision=precision, use_torch_compile=compile_enabled)
