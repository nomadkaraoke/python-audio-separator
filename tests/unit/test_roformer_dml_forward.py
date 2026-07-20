"""DML CPU-hop equivalence tests for the Roformer forward passes (issue #292).

torch-directml has no complex tensor support, so bs_roformer/mel_band_roformer
route their complex ops (stft, view_as_complex, complex multiply, istft) to CPU
when running on a privateuseone device. We can't create DML tensors in CI, but
we CAN force the DML branches on CPU tensors (where every hop is a no-op) and
assert the output is identical to the untouched path — proving the hop plumbing
itself doesn't alter results, reorder dims, or drop tensors.
"""

import pytest
import torch
from unittest.mock import patch

from audio_separator.separator.uvr_lib_v5.roformer import bs_roformer as bs_mod
from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mel_mod


def _tiny_bs_roformer():
    torch.manual_seed(0)
    return bs_mod.BSRoformer(
        dim=32,
        depth=1,
        stereo=False,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        freqs_per_bands=(129, 128),  # sums to 512 // 2 + 1
        stft_n_fft=512,
        stft_hop_length=128,
        stft_win_length=512,
    ).eval()


def _tiny_mel_band_roformer():
    torch.manual_seed(0)
    return mel_mod.MelBandRoformer(
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


class TestIsDmlDeviceHelper:
    def test_cpu_is_not_dml(self):
        assert not bs_mod._is_dml_device(torch.device("cpu"))
        assert not mel_mod._is_dml_device(torch.device("cpu"))

    def test_privateuseone_is_dml(self):
        assert bs_mod._is_dml_device(torch.device("privateuseone", 0))
        assert mel_mod._is_dml_device(torch.device("privateuseone", 0))


class TestBSRoformerDmlBranchEquivalence:
    def test_forced_dml_branch_matches_normal_cpu_output(self):
        model = _tiny_bs_roformer()
        torch.manual_seed(1)
        audio = torch.randn(1, 8192)

        with torch.no_grad():
            normal = model(audio)
            with patch.object(bs_mod, "_is_dml_device", return_value=True):
                hopped = model(audio)

        assert hopped.device == audio.device
        assert hopped.shape == normal.shape
        assert torch.allclose(normal, hopped, atol=1e-6), "DML CPU-hop branch changed the output"


class TestMelBandRoformerDmlBranchEquivalence:
    def test_forced_dml_branch_matches_normal_cpu_output(self):
        model = _tiny_mel_band_roformer()
        torch.manual_seed(1)
        audio = torch.randn(1, 8192)

        with torch.no_grad():
            normal = model(audio)
            with patch.object(mel_mod, "_is_dml_device", return_value=True):
                hopped = model(audio)

        assert hopped.device == audio.device
        assert hopped.shape == normal.shape
        assert torch.allclose(normal, hopped, atol=1e-6), "DML CPU-hop branch changed the output"


class TestAttendDmlFallback:
    """SDPA is unimplemented on torch-directml — Attend must route DML tensors
    to the einsum math path. Forcing the gate on CPU proves the fallback path
    produces the same attention output as SDPA."""

    def test_is_dml_device(self):
        from audio_separator.separator.uvr_lib_v5.roformer import attend as attend_mod

        assert not attend_mod._is_dml_device(torch.device("cpu"))
        assert attend_mod._is_dml_device(torch.device("privateuseone", 0))

    def test_math_fallback_matches_sdpa_output(self):
        from audio_separator.separator.uvr_lib_v5.roformer import attend as attend_mod

        torch.manual_seed(0)
        att = attend_mod.Attend(flash=True)
        att.eval()
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)

        with torch.no_grad():
            flash_out = att(q, k, v)
            with patch.object(attend_mod, "_is_dml_device", return_value=True):
                math_out = att(q, k, v)

        assert math_out.shape == flash_out.shape
        assert torch.allclose(flash_out, math_out, atol=1e-5), "einsum fallback diverges from SDPA"

    def test_sliced_attention_matches_unsliced(self):
        # The DML path slices the batch dim (step=8) to bound peak memory;
        # use a batch that is neither a multiple of the step nor smaller than
        # it, to cover the ragged final slice.
        from audio_separator.separator.uvr_lib_v5.roformer import attend as attend_mod

        torch.manual_seed(0)
        att = attend_mod.Attend(flash=False)
        att.eval()
        q = torch.randn(19, 4, 16, 32)
        k = torch.randn(19, 4, 16, 32)
        v = torch.randn(19, 4, 16, 32)

        with torch.no_grad():
            unsliced = att(q, k, v)
            with patch.object(attend_mod, "_is_dml_device", return_value=True):
                sliced = att(q, k, v)

        assert torch.allclose(unsliced, sliced, atol=1e-6), "batch-sliced attention diverges"


class TestRotaryNoCatEquivalence:
    """The DML rotary path computes t*cos + rotate_half(t)*sin directly,
    skipping the (empty) edge concat torch-directml rejects. It must match
    the library implementation exactly."""

    def test_manual_rotation_matches_library(self):
        from rotary_embedding_torch import RotaryEmbedding

        torch.manual_seed(0)
        rotary = RotaryEmbedding(dim=64)
        t = torch.randn(2, 8, 16, 64)

        library = rotary.rotate_queries_or_keys(t)
        with patch.object(bs_mod, "_is_dml_device", return_value=True):
            manual_bs = bs_mod._rotate_queries_or_keys(rotary, t)
        with patch.object(mel_mod, "_is_dml_device", return_value=True):
            manual_mel = mel_mod._rotate_queries_or_keys(rotary, t)

        assert torch.allclose(library, manual_bs, atol=1e-6)
        assert torch.allclose(library, manual_mel, atol=1e-6)
