from unittest.mock import Mock, patch

import pytest
import torch

from audio_separator.separator.uvr_lib_v5 import device_utils
from audio_separator.separator.uvr_lib_v5.stft import STFT as CommonSTFT
from audio_separator.separator.uvr_lib_v5.tfc_tdf_v3 import STFT as TFCSTFT


def _common_stft(device):
    return CommonSTFT(Mock(), n_fft=256, hop_length=64, dim_f=129, device=device)


def _tfc_stft(device):
    return TFCSTFT(n_fft=256, hop_length=64, dim_f=129, device=device)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.filterwarnings("ignore:stft with return_complex=False is deprecated:UserWarning")
@pytest.mark.parametrize(
    ("factory", "module_name"),
    [
        pytest.param(
            _common_stft,
            "audio_separator.separator.uvr_lib_v5.stft",
            id="common-stft",
        ),
        pytest.param(
            _tfc_stft,
            "audio_separator.separator.uvr_lib_v5.tfc_tdf_v3",
            id="tfc-tdf-stft",
        ),
    ],
)
@pytest.mark.parametrize("force_cpu_fallback", [False, True])
def test_mps_stft_round_trip_uses_selected_compute_device(
    factory,
    module_name,
    force_cpu_fallback,
    monkeypatch,
):
    monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", raising=False)
    mps_device = torch.device("mps")

    if not force_cpu_fallback and device_utils.should_fallback_to_cpu_for_complex_ops(mps_device):
        pytest.skip("This MPS runtime does not support the native spectral path")

    indices = torch.arange(2048, dtype=torch.float32)
    audio = torch.stack(
        (
            0.5 * torch.sin(2 * torch.pi * 440 * indices / 44100),
            0.3 * torch.cos(2 * torch.pi * 880 * indices / 44100),
        )
    ).unsqueeze(0)

    cpu_stft = factory(torch.device("cpu"))
    reference_spectrum = cpu_stft(audio)
    reference_audio = cpu_stft.inverse(reference_spectrum)

    real_stft = torch.stft
    real_istft = torch.istft
    stft_devices = []
    istft_devices = []

    def tracked_stft(input_tensor, *args, **kwargs):
        stft_devices.append(input_tensor.device.type)
        return real_stft(input_tensor, *args, **kwargs)

    def tracked_istft(input_tensor, *args, **kwargs):
        istft_devices.append(input_tensor.device.type)
        return real_istft(input_tensor, *args, **kwargs)

    with (
        patch(
            f"{module_name}.should_fallback_to_cpu_for_complex_ops",
            return_value=force_cpu_fallback,
        ) as fallback,
        patch(f"{module_name}.torch.stft", side_effect=tracked_stft),
        patch(f"{module_name}.torch.istft", side_effect=tracked_istft),
    ):
        mps_stft = factory(mps_device)
        spectrum = mps_stft(audio.to(mps_device))
        reconstructed = mps_stft.inverse(spectrum)

    compute_device = "cpu" if force_cpu_fallback else "mps"
    assert stft_devices == [compute_device]
    assert istft_devices == [compute_device]
    assert spectrum.device.type == "mps"
    assert reconstructed.device.type == "mps"
    assert fallback.call_count == 2

    torch.testing.assert_close(
        spectrum.cpu(),
        reference_spectrum,
        rtol=2e-4,
        atol=2e-5,
    )
    torch.testing.assert_close(
        reconstructed.cpu(),
        reference_audio,
        rtol=2e-4,
        atol=2e-5,
    )
