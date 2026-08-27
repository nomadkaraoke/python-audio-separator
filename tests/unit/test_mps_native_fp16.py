import copy
import platform
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from rotary_embedding_torch import RotaryEmbedding

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.execution_policy import AUTOCAST, FP32, NATIVE_FP16
from audio_separator.separator.separator import Separator
from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mel_module


def _apple_gpu_is_virtualized() -> bool:
    """Detect a paravirtualized Metal device (hosted CI Macs report VirtualMac*)."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.model"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return result.stdout.strip().startswith("VirtualMac")


class MelBandRoformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.rotary_embedding = RotaryEmbedding(dim=8)


def _separator(device="mps", use_native_fp16=True, model_type="mel_band_roformer"):
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_run = MelBandRoformer()
    separator.roformer_model_type = model_type
    separator.torch_device = torch.device(device)
    separator.requested_torch_device = separator.torch_device
    separator.use_autocast = False
    separator.use_native_fp16 = use_native_fp16
    separator.use_torch_compile = False
    separator.is_native_fp16 = False
    return separator


def _dispatch_separator(*, effective_precision, device="mps", separator_device=None):
    separator = object.__new__(Separator)
    separator.chunk_duration = None
    separator.logger = Mock()
    separator.normalization_threshold = 1.0
    separator.amplification_threshold = 0.0
    separator.use_autocast = effective_precision == AUTOCAST
    separator.use_native_fp16 = effective_precision == NATIVE_FP16
    separator.torch_device = torch.device(separator_device or device)
    separator.model_instance = Mock(effective_precision=effective_precision, torch_device=torch.device(device))
    separator.model_instance.separate.return_value = ["output.wav"]
    separator.print_uvr_vip_message = Mock()
    return separator


def test_mel_band_roformer_uses_explicit_native_fp16_on_mps():
    separator = _separator()
    separator.model_run.rotary_embedding.rotate_queries_or_keys(torch.randn(1, 4, 8))
    expected_frequencies = separator.model_run.rotary_embedding.freqs.detach().clone()
    assert separator.model_run.rotary_embedding.cached_freqs is not None

    separator._configure_model_precision()

    assert separator.is_native_fp16 is True
    assert separator.model_run.projection.weight.dtype == torch.float16
    assert separator.model_run.rotary_embedding.freqs.dtype == torch.float32
    assert separator.model_run.rotary_embedding.cached_freqs is None
    torch.testing.assert_close(separator.model_run.rotary_embedding.freqs, expected_frequencies, rtol=0, atol=0)

    output = separator.model_run.rotary_embedding.rotate_queries_or_keys(torch.randn(1, 1101, 8).half())

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert separator.model_run.rotary_embedding.cached_freqs.dtype == torch.float32


@pytest.mark.parametrize(
    ("device", "use_native_fp16", "model_type"),
    [
        ("cpu", True, "mel_band_roformer"),
        ("mps", False, "mel_band_roformer"),
    ],
)
def test_native_fp16_requires_verified_device_and_model(device, use_native_fp16, model_type):
    separator = _separator(device=device, use_native_fp16=use_native_fp16, model_type=model_type)

    separator._configure_model_precision()

    assert separator.is_native_fp16 is False
    assert separator.model_run.projection.weight.dtype == torch.float32


def test_native_fp16_supports_legacy_loader_class_detection():
    separator = _separator(model_type=None)

    separator._configure_model_precision()

    assert separator.is_native_fp16 is True


@pytest.mark.parametrize(
    ("device", "use_native_fp16", "expected_load_device"),
    [
        ("mps", True, "cpu"),
        ("mps", False, "mps"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
        ("privateuseone", True, "privateuseone"),
    ],
)
def test_roformer_loads_on_cpu_only_before_native_mps_conversion(device, use_native_fp16, expected_load_device):
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.is_roformer = True
    separator.model_data = {}
    separator.model_path = "/tmp/model.ckpt"
    separator.torch_device = torch.device(device)
    separator.requested_torch_device = separator.torch_device
    separator.use_autocast = False
    separator.use_native_fp16 = use_native_fp16
    separator.use_torch_compile = False
    loaded_model = Mock()
    loaded_model.to.return_value = loaded_model
    separator.roformer_loader = Mock(
        load_model=Mock(
            return_value=SimpleNamespace(
                success=True,
                model=loaded_model,
                model_info={"model_type": "mel_band_roformer"},
            )
        )
    )

    with (
        patch.object(separator, "_configure_model_precision") as configure_precision,
        patch.object(separator, "_configure_model_compilation") as configure_compilation,
    ):
        separator.load_model()

    separator.roformer_loader.load_model.assert_called_once_with(
        model_path="/tmp/model.ckpt",
        config={},
        device=expected_load_device,
    )
    configure_precision.assert_called_once_with()
    configure_compilation.assert_called_once_with()
    loaded_model.to.assert_called_once_with(torch.device(device))
    loaded_model.eval.assert_called_once_with()


def test_separator_bypasses_autocast_for_native_fp16():
    separator = _dispatch_separator(effective_precision=NATIVE_FP16)

    with (
        patch("audio_separator.separator.separator.autocast_mode.autocast", return_value=nullcontext()) as autocast,
    ):
        output_files = separator._separate_file("input.wav")

    assert output_files == ["output.wav"]
    autocast.assert_not_called()
    separator.logger.debug.assert_any_call("Using native float16 inference.")


def test_separator_preserves_non_native_autocast():
    separator = _dispatch_separator(effective_precision=AUTOCAST, device="cpu", separator_device="mps")

    with (
        patch("audio_separator.separator.separator.autocast_mode.autocast", return_value=nullcontext()) as autocast,
    ):
        separator._separate_file("input.wav")

    autocast.assert_called_once_with("cpu")


def test_separator_uses_float32_for_fallback_policy():
    separator = _dispatch_separator(effective_precision=FP32, device="cpu")

    with patch("audio_separator.separator.separator.autocast_mode.autocast", return_value=nullcontext()) as autocast:
        separator._separate_file("input.wav")

    autocast.assert_not_called()
    separator.logger.debug.assert_any_call("Using float32 inference.")


def _tiny_mel_band_roformer():
    torch.manual_seed(0)
    return mel_module.MelBandRoformer(
        dim=32,
        depth=1,
        stereo=False,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=8,
        stft_n_fft=512,
        stft_hop_length=128,
        stft_win_length=512,
    ).eval()


def _half_preserving_rotary_frequencies(model):
    rotary_frequencies = [module.freqs.detach().clone() for module in model.modules() if isinstance(module, RotaryEmbedding)]

    model.half()

    rotary_modules = [module for module in model.modules() if isinstance(module, RotaryEmbedding)]
    for rotary, frequencies in zip(rotary_modules, rotary_frequencies):
        rotary.freqs.data = frequencies.to(rotary.freqs.device)
        rotary.cached_freqs = None

    return model


def test_mel_band_half_forward_keeps_cpu_silence_finite_and_zero():
    model = _half_preserving_rotary_frequencies(_tiny_mel_band_roformer())

    with torch.no_grad():
        output = model(torch.zeros(1, 8192))

    assert output.shape == (1, 1, 8192)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.parametrize("force_cpu_complex", [False, True])
def test_native_fp16_mps_forward_handles_silence(force_cpu_complex):
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_run = _tiny_mel_band_roformer()
    separator.roformer_model_type = "mel_band_roformer"
    separator.torch_device = torch.device("mps")
    separator.requested_torch_device = separator.torch_device
    separator.use_autocast = False
    separator.use_native_fp16 = True
    separator.use_torch_compile = False
    separator.is_native_fp16 = False
    separator._configure_model_precision()
    separator.model_run.to(separator.torch_device)
    audio = torch.zeros(1, 8192, device=separator.torch_device)

    with patch.object(mel_module, "should_fallback_to_cpu_for_complex_ops", return_value=force_cpu_complex), torch.no_grad():
        output = separator.model_run(audio)

    assert output.shape == audio.unsqueeze(1).shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
    assert output.device.type == ("cpu" if force_cpu_complex else "mps")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.skipif(
    _apple_gpu_is_virtualized(),
    reason="fp16 SNR gates need a real Apple GPU: virtualized Metal degrades half-precision accumulation",
)
@pytest.mark.parametrize("force_cpu_complex", [False, True])
def test_native_fp16_mps_forward_matches_cpu_fp32_for_non_silent_audio(force_cpu_complex):
    cpu_model = _tiny_mel_band_roformer()
    mps_model = copy.deepcopy(cpu_model)
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_run = mps_model
    separator.roformer_model_type = "mel_band_roformer"
    separator.torch_device = torch.device("mps")
    separator.requested_torch_device = separator.torch_device
    separator.use_autocast = False
    separator.use_native_fp16 = True
    separator.use_torch_compile = False
    separator.is_native_fp16 = False
    separator._configure_model_precision()
    separator.model_run.to(separator.torch_device)

    sample_indices = torch.arange(8192, dtype=torch.float32)
    audio = (
        0.35 * torch.sin(2 * torch.pi * 440 * sample_indices / 44100) + 0.15 * torch.sin(2 * torch.pi * 880 * sample_indices / 44100)
    ).unsqueeze(0)

    with torch.no_grad():
        reference = cpu_model(audio)
        with patch.object(mel_module, "should_fallback_to_cpu_for_complex_ops", return_value=force_cpu_complex):
            output = separator.model_run(audio.to(separator.torch_device))

    assert output.device.type == ("cpu" if force_cpu_complex else "mps")
    output = output.cpu().float()
    error = output - reference
    snr = 20 * torch.log10(reference.square().mean().sqrt() / error.square().mean().sqrt())

    assert output.shape == reference.shape
    assert torch.isfinite(output).all()
    assert reference.square().mean().sqrt().item() > 1e-4
    assert snr.item() > 30
    torch.testing.assert_close(output, reference, rtol=0.1, atol=1e-3)
