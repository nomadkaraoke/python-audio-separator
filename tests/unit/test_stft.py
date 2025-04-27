import unittest
import numpy as np
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
        self.n_fft = 2048
        self.hop_length = 512
        self.dim_f = 1025
        self.device = torch.device("cpu")
        self.stft = STFT(logger=Mock(), n_fft=self.n_fft, hop_length=self.hop_length, dim_f=self.dim_f, device=self.device)

    def create_mock_tensor(self, shape, device=None):
        tensor = torch.rand(shape)
        if device:
            tensor = tensor.to(device)
        return tensor

    def test_stft_initialization(self):
        self.assertEqual(self.stft.n_fft, self.n_fft)
        self.assertEqual(self.stft.hop_length, self.hop_length)
        self.assertEqual(self.stft.dim_f, self.dim_f)
        self.assertEqual(self.stft.device.type, "cpu")
        self.assertIsInstance(self.stft.hann_window, torch.Tensor)

    def test_stft_call(self):
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
        mps_device = torch.device("mps")
        self.stft.device = mps_device
        input_tensor = self.create_mock_tensor((1, 16000), device=mps_device)
        stft_result = self.stft(input_tensor)
        self.assertIsNotNone(stft_result)
        self.assertIsInstance(stft_result, torch.Tensor)

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_inverse_with_mps_device(self):
        mps_device = torch.device("mps")
        self.stft.device = mps_device
        input_tensor = self.create_mock_tensor((1, 2, 1025, 32), device=mps_device)
        istft_result = self.stft.inverse(input_tensor)
        self.assertIsNotNone(istft_result)
        self.assertIsInstance(istft_result, torch.Tensor)


# Mock logger to use in tests
class MockLogger:
    def debug(self, message):
        pass


if __name__ == "__main__":
    unittest.main()
