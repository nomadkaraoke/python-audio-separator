import unittest
import numpy as np
import pytest
import torch
from unittest.mock import Mock
from audio_separator.separator.uvr_lib_v5.stft import STFT

# Short-Time Fourier Transform (STFT) Process Overview:
#
# STFT transforms a time-domain signal into a frequency-domain representation.
#   This transformation is achieved by dividing the signal into short frames (or segments) and applying the Fourier Transform to each frame.
#
# n_fft: The number of points used in the Fourier Transform, which determines the resolution of the frequency domain representation.
#   Essentially, it dictates how many frequency bins we get in our STFT.
#
# hop_length: The number of samples by which we shift each frame of the signal.
#   It affects the overlap between consecutive frames. If the hop_length is less than n_fft, we get overlapping frames.
#
# Windowing: Each frame of the signal is multiplied by a window function (e.g. Hann window) before applying the Fourier Transform.
#   This is done to minimize discontinuities at the borders of each frame.


class TestSTFT(unittest.TestCase):
    def setUp(self):
        """Configure a CPU STFT instance with a complete frequency spectrum."""
        self.n_fft = 2048
        self.hop_length = 512
        self.dim_f = 1025
        self.device = torch.device("cpu")
        self.stft = STFT(logger=Mock(), n_fft=self.n_fft, hop_length=self.hop_length, dim_f=self.dim_f, device=self.device)

    def create_mock_tensor(self, shape, device=None):
        """Create random test input on the requested device."""
        tensor = torch.rand(shape)
        if device:
            tensor = tensor.to(device)
        return tensor

    def test_stft_initialization(self):
        """Retain the configured transform dimensions, device and Hann window."""
        self.assertEqual(self.stft.n_fft, self.n_fft)
        self.assertEqual(self.stft.hop_length, self.hop_length)
        self.assertEqual(self.stft.dim_f, self.dim_f)
        self.assertEqual(self.stft.device.type, "cpu")
        self.assertIsInstance(self.stft.hann_window, torch.Tensor)

    def test_stft_call(self):
        """Produce the expected frequency-bin and time-frame dimensions."""
        input_tensor = self.create_mock_tensor((1, 16000))

        # Apply STFT
        stft_result = self.stft(input_tensor)

        # Test conditions
        self.assertIsNotNone(stft_result)
        self.assertIsInstance(stft_result, torch.Tensor)

        # Calculate the expected shape based on input parameters:

        # Frequency Dimension (dim_f): This corresponds to the number of frequency bins in the STFT output.
        #   In the case of a real-valued input signal (like audio), the Fourier Transform produces a symmetric output.
        #   Hence, for an n_fft of 2048, we would typically get 2049 frequency bins (from 0 Hz to the Nyquist frequency).
        #   However, we often don't need the full symmetric spectrum.
        #   So, dim_f is used to specify how many frequency bins we are interested in.
        #   In this test, it's set to 1025, which is about half of n_fft + 1 (as the Fourier Transform of a real-valued signal is symmetric).

        # Time Dimension: This corresponds to how many frames (or segments) the input signal has been divided into.
        #   It depends on the length of the input signal and the hop_length.
        #   The formula for calculating the number of frames is derived from how we stride the window across the signal:
        #     Length of Input Signal: Let's denote it as L. In this test, the input tensor has a shape of [1, 16000], so L is 16000 (ignoring the batch dimension for simplicity).
        #     Number of Frames: The number of frames depends on how we stride the window across the signal. For each frame, we move the window by hop_length samples.
        #     Therefore, the number of frames N_frames can be roughly estimated by dividing the length of the signal by the hop_length.
        #     However, since the window overlaps the signal, we add an extra frame to account for the last segment of the signal. This gives us N_frames = (L // hop_length) + 1.

        # Putting It All Together
        #   expected_shape thus becomes (dim_f, N_frames), which is (1025, (16000 // 512) + 1) in this test case.

        expected_shape = (self.dim_f, (input_tensor.shape[1] // self.hop_length) + 1)

        self.assertEqual(stft_result.shape[-2:], expected_shape)

    def test_calculate_inverse_dimensions(self):
        """Recover inverse-transform dimensions from the packed spectrum shape."""
        # Create a sample input tensor
        sample_input = torch.randn(1, 2, 500, 32)  # Batch, Channel, Frequency, Time dimensions
        batch_dims, channel_dim, freq_dim, time_dim, num_freq_bins = self.stft.calculate_inverse_dimensions(sample_input)

        # Expected values
        expected_num_freq_bins = self.n_fft // 2 + 1

        # Assertions
        self.assertEqual(batch_dims, sample_input.shape[:-3])
        self.assertEqual(channel_dim, 2)
        self.assertEqual(freq_dim, 500)
        self.assertEqual(time_dim, 32)
        self.assertEqual(num_freq_bins, expected_num_freq_bins)

    def test_pad_frequency_dimension(self):
        """Extend a truncated spectrum to the full inverse-transform frequency dimension."""
        # Create a sample input tensor
        sample_input = torch.randn(1, 2, 500, 32)  # Batch, Channel, Frequency, Time dimensions
        batch_dims, channel_dim, freq_dim, time_dim, num_freq_bins = self.stft.calculate_inverse_dimensions(sample_input)

        # Apply padding
        padded_output = self.stft.pad_frequency_dimension(sample_input, batch_dims, channel_dim, freq_dim, time_dim, num_freq_bins)

        # Expected frequency dimension after padding
        expected_freq_dim = num_freq_bins

        # Assertions
        self.assertEqual(padded_output.shape[-2], expected_freq_dim)

    def test_prepare_for_istft(self):
        """Combine packed real and imaginary planes into the expected complex spectrum shape."""
        # Create a sample input tensor
        sample_input = torch.randn(1, 2, 500, 32)  # Batch, Channel, Frequency, Time dimensions
        batch_dims, channel_dim, freq_dim, time_dim, num_freq_bins = self.stft.calculate_inverse_dimensions(sample_input)
        padded_output = self.stft.pad_frequency_dimension(sample_input, batch_dims, channel_dim, freq_dim, time_dim, num_freq_bins)

        # Apply prepare_for_istft
        complex_tensor = self.stft.prepare_for_istft(padded_output, batch_dims, channel_dim, num_freq_bins, time_dim)

        # Calculate the expected flattened batch size (flattening batch and channel dimensions)
        expected_flattened_batch_size = batch_dims[0] * (channel_dim // 2)

        # Expected shape of the complex tensor
        expected_shape = (expected_flattened_batch_size, num_freq_bins, time_dim)

        # Assertions
        self.assertEqual(complex_tensor.shape, expected_shape)

    def test_inverse_stft(self):
        """Preserve the legacy inverse output shape on CPU."""
        # Create a mock tensor with the correct input shape
        input_tensor = torch.rand(1, 2, 1025, 32)  # shape matching output of STFT

        # Apply inverse STFT
        output_tensor = self.stft.inverse(input_tensor)

        # Check if the output tensor is on the CPU
        self.assertEqual(output_tensor.device.type, "cpu")

        # Expected output shape: (Batch size, Channel dimension, Time dimension)
        expected_shape = (1, 2, 7936)  # Calculated based on STFT parameters

        # Check if the output tensor has the expected shape
        self.assertEqual(output_tensor.shape, expected_shape)

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_stft_with_mps_device(self):
        """Transform input on an available MPS device."""
        mps_device = torch.device("mps")
        self.stft.device = mps_device
        input_tensor = self.create_mock_tensor((1, 16000), device=mps_device)
        stft_result = self.stft(input_tensor)
        self.assertIsNotNone(stft_result)
        self.assertIsInstance(stft_result, torch.Tensor)

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_inverse_with_mps_device(self):
        """Reconstruct audio from spectra supplied on an available MPS device."""
        mps_device = torch.device("mps")
        self.stft.device = mps_device
        input_tensor = self.create_mock_tensor((1, 2, 1025, 32), device=mps_device)
        istft_result = self.stft.inverse(input_tensor)
        self.assertIsNotNone(istft_result)
        self.assertIsInstance(istft_result, torch.Tensor)


# Mock logger to use in tests
class MockLogger:
    def debug(self, message):
        """Accept debug messages without emitting output during STFT tests."""
        pass


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("force_cpu_fallback", [False, True])
def test_stft_preserves_mdx_channel_order_and_round_trip(device, force_cpu_fallback, monkeypatch):
    """Preserve channel packing, reconstruction and window reuse through native or CPU fallback paths."""
    if force_cpu_fallback:
        monkeypatch.setenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", "1")
    else:
        monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", raising=False)
    generator = torch.Generator().manual_seed(17)
    audio = torch.randn(3, 2, 2048, generator=generator) * 0.1
    transform = STFT(Mock(), n_fft=256, hop_length=64, dim_f=129, device=torch.device(device))
    window = torch.hann_window(256)
    complex_spectrum = torch.stft(audio.reshape(-1, 2048), n_fft=256, hop_length=64, window=window, return_complex=True)
    complex_spectrum = complex_spectrum.reshape(3, 2, 129, -1)
    expected = torch.stack(
        [complex_spectrum[:, 0].real, complex_spectrum[:, 0].imag, complex_spectrum[:, 1].real, complex_spectrum[:, 1].imag], dim=1
    )
    actual = transform(audio.to(device))
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    window_pointer = transform.hann_window.data_ptr()
    restored = transform.inverse(actual)
    torch.testing.assert_close(restored.cpu(), audio, rtol=2e-5, atol=2e-6)
    assert restored.device.type == device
    assert transform.hann_window.device.type == ("cpu" if force_cpu_fallback else device)
    assert transform.hann_window.data_ptr() == window_pointer
    transform(audio.to(device))
    assert transform.hann_window.data_ptr() == window_pointer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frequency_padding_preserves_dtype_and_zeroes_omitted_bins(dtype):
    """Preserve spectral values and dtype while zero-filling omitted frequency bins."""
    transform = STFT(Mock(), n_fft=256, hop_length=64, dim_f=100, device=torch.device("cpu"))
    spectrum = torch.randn(3, 4, 100, 17, dtype=dtype)
    dimensions = transform.calculate_inverse_dimensions(spectrum)
    padded = transform.pad_frequency_dimension(spectrum, *dimensions)
    assert padded.dtype == dtype
    torch.testing.assert_close(padded[..., :100, :], spectrum)
    assert torch.count_nonzero(padded[..., 100:, :]) == 0
    actual = transform.prepare_for_istft(padded, dimensions[0], dimensions[1], dimensions[-1], dimensions[-2])
    real, imaginary = padded.reshape(-1, 2, 129, 17).unbind(dim=1)
    torch.testing.assert_close(actual, torch.complex(real, imaginary), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
