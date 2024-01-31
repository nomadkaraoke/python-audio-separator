import unittest
import numpy as np
from audio_separator.separator.spec_utils import crop_center, preprocess, make_padding, wave_to_spectrogram, wave_to_spectrogram_mt


class TestSpecUtils(unittest.TestCase):
    def test_preprocess(self):
        X_spec = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
        X_mag, X_phase = preprocess(X_spec)
        self.assertEqual(X_mag.shape, X_spec.shape)
        self.assertEqual(X_phase.shape, X_spec.shape)

    def test_make_padding(self):
        width, cropsize, offset = 100, 50, 10
        left, right, roi_size = make_padding(width, cropsize, offset)
        self.assertEqual(left, 10)
        self.assertTrue(right >= left)
        self.assertEqual(roi_size, 30)

    def test_preprocess_values(self):
        X_spec = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
        X_mag, X_phase = preprocess(X_spec)
        self.assertTrue((X_mag >= 0).all())
        self.assertTrue((X_phase >= -np.pi).all() and (X_phase <= np.pi).all())

    def test_make_padding_values(self):
        width, cropsize, offset = 100, 50, 10
        left, right, roi_size = make_padding(width, cropsize, offset)
        self.assertTrue(left >= 0)
        self.assertTrue(right >= 0)
        self.assertTrue(roi_size > 0)

    def test_preprocess_magnitude_phase(self):
        X_spec = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        X_mag, X_phase = preprocess(X_spec)
        self.assertTrue(np.all(X_mag >= 0))
        self.assertTrue(np.all(X_phase >= -np.pi) and np.all(X_phase <= np.pi))


if __name__ == "__main__":
    unittest.main()
