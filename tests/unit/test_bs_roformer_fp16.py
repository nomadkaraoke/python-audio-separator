import copy
from unittest.mock import Mock, patch

import pytest
import torch
from rotary_embedding_torch import RotaryEmbedding

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.execution_policy import NATIVE_FP16
from audio_separator.separator.uvr_lib_v5.roformer import bs_roformer as bs_module


def _tiny_bs_roformer(*, linear_transformer_depth=0):
    torch.manual_seed(0)
    return bs_module.BSRoformer(
        dim=16,
        depth=1,
        stereo=False,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        linear_transformer_depth=linear_transformer_depth,
        freqs_per_bands=(17, 16),  # sums to 64 // 2 + 1
        dim_head=8,
        heads=2,
        flash_attn=False,
        stft_n_fft=64,
        stft_hop_length=16,
        stft_win_length=64,
        mask_estimator_depth=1,
    ).eval()


def _half_preserving_rotary_frequencies(model):
    rotary_frequencies = [module.freqs.detach().clone() for module in model.modules() if isinstance(module, RotaryEmbedding)]

    model.half()

    rotary_modules = [module for module in model.modules() if isinstance(module, RotaryEmbedding)]
    for rotary, frequencies in zip(rotary_modules, rotary_frequencies):
        rotary.freqs.data = frequencies.to(rotary.freqs.device)
        rotary.cached_freqs = None

    return model


def _non_silent_audio():
    sample_indices = torch.arange(256, dtype=torch.float32)
    return (
        0.35 * torch.sin(2 * torch.pi * 440 * sample_indices / 44100) + 0.15 * torch.sin(2 * torch.pi * 880 * sample_indices / 44100)
    ).unsqueeze(0)


def test_bs_rms_norm_keeps_half_silence_finite_and_zero_on_cpu():
    norm = bs_module.RMSNorm(16).half()

    output = norm(torch.zeros(2, 4, 16, dtype=torch.float16))

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_bs_linear_attention_normalization_keeps_half_silence_finite_on_cpu():
    output = bs_module.l2norm(torch.zeros(2, 4, 16, dtype=torch.float16))

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("flash", [False, True])
def test_bs_linear_attention_uses_its_configured_similarity_scale(flash):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    scale = 8.0
    attend = bs_module.Attend(scale=scale, flash=flash)

    similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    expected = torch.einsum("b h i j, b h j d -> b h i d", similarity.softmax(dim=-1), v)

    torch.testing.assert_close(attend(q, k, v), expected)


def test_bs_half_forward_keeps_silence_finite_and_zero_on_cpu():
    model = _half_preserving_rotary_frequencies(_tiny_bs_roformer())

    with torch.no_grad():
        output = model(torch.zeros(1, 256))

    assert output.shape == (1, 1, 256)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_bs_native_fp16_is_selected_and_applied_on_mps():
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_run = _tiny_bs_roformer()
    separator.roformer_model_type = "bs_roformer"
    separator.torch_device = torch.device("mps")
    separator.requested_torch_device = separator.torch_device
    separator.use_autocast = False
    separator.use_native_fp16 = True
    separator.use_torch_compile = False

    separator._configure_model_precision()

    assert separator.effective_precision == NATIVE_FP16
    assert separator.is_native_fp16 is True
    assert next(separator.model_run.band_split.parameters()).dtype == torch.float16


def test_bs_half_forward_keeps_silence_finite_in_forced_dml_complex_fallback():
    model = _half_preserving_rotary_frequencies(_tiny_bs_roformer())

    with patch.object(bs_module, "_is_dml_device", return_value=True), torch.no_grad():
        output = model(torch.zeros(1, 256))

    assert output.shape == (1, 1, 256)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_regional_compile_target_flattening_handles_two_and_three_transformer_bs_blocks():
    separator = object.__new__(MDXCSeparator)
    two_transformer_block = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    three_transformer_block = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()])
    separator.model_run = torch.nn.Module()
    separator.model_run.layers = torch.nn.ModuleList([two_transformer_block, three_transformer_block])

    targets = separator._regional_compile_targets()

    assert targets == [*two_transformer_block, *three_transformer_block]


def test_bs_linear_attention_block_runs_in_half_and_is_included_in_regional_compile():
    model = _half_preserving_rotary_frequencies(_tiny_bs_roformer(linear_transformer_depth=1))
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_run = model
    separator.roformer_model_type = "bs_roformer"
    separator.torch_device = torch.device("cpu")
    separator._should_torch_compile = True

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert len(model.layers[0]) == 3
    assert compile_module.call_count == 3
    assert separator.effective_torch_compile is True

    with torch.no_grad():
        output = model(torch.zeros(1, 256))

    assert output.shape == (1, 1, 256)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.parametrize("force_cpu_complex", [False, True])
def test_bs_half_mps_forward_keeps_silence_finite_and_zero(force_cpu_complex):
    model = _half_preserving_rotary_frequencies(_tiny_bs_roformer()).to("mps")
    audio = torch.zeros(1, 256, device="mps")
    original_stft = torch.stft

    with (
        patch.object(bs_module, "should_fallback_to_cpu_for_complex_ops", return_value=force_cpu_complex),
        patch.object(bs_module.torch, "stft", wraps=original_stft) as stft,
        torch.no_grad(),
    ):
        output = model(audio)

    assert output.shape == (1, 1, 256)
    assert output.device.type == "mps"
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
    assert stft.call_args.args[0].device.type == ("cpu" if force_cpu_complex else "mps")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.parametrize("force_cpu_complex", [False, True])
def test_bs_half_mps_forward_matches_cpu_float32(force_cpu_complex):
    cpu_model = _tiny_bs_roformer()
    mps_model = _half_preserving_rotary_frequencies(copy.deepcopy(cpu_model)).to("mps")
    audio = _non_silent_audio()

    with torch.no_grad():
        reference = cpu_model(audio)
        with patch.object(bs_module, "should_fallback_to_cpu_for_complex_ops", return_value=force_cpu_complex):
            output = mps_model(audio.to("mps"))

    output = output.cpu().float()
    error = output - reference
    reference_rms = reference.square().mean().sqrt()
    error_rms = error.square().mean().sqrt()
    snr = 20 * torch.log10(reference_rms / error_rms)

    assert output.shape == reference.shape
    assert torch.isfinite(output).all()
    assert reference_rms.item() > 1e-4
    assert snr.item() > 30
    torch.testing.assert_close(output, reference, rtol=0.1, atol=1e-3)
