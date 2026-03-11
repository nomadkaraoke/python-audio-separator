import pytest
import numpy as np
import logging
from audio_separator.separator.ensembler import Ensembler

@pytest.fixture
def logger():
    return logging.getLogger("test")

def test_ensembler_avg_wave(logger):
    # Test simple averaging
    wav1 = np.ones((2, 100))
    wav2 = np.zeros((2, 100))
    ensembler = Ensembler(logger, algorithm="avg_wave")
    result = ensembler.ensemble([wav1, wav2])
    assert np.allclose(result, 0.5)

def test_ensembler_weighted_avg(logger):
    # Test weighted averaging
    wav1 = np.ones((2, 100))
    wav2 = np.zeros((2, 100))
    ensembler = Ensembler(logger, algorithm="avg_wave", weights=[3.0, 1.0])
    result = ensembler.ensemble([wav1, wav2])
    assert np.allclose(result, 0.75)

def test_ensembler_different_lengths(logger):
    # Test padding for different lengths
    wav1 = np.ones((2, 100))
    wav2 = np.zeros((2, 80))
    ensembler = Ensembler(logger, algorithm="avg_wave")
    result = ensembler.ensemble([wav1, wav2])
    assert result.shape == (2, 100)
    assert np.allclose(result[:, :80], 0.5)
    assert np.allclose(result[:, 80:], 0.5) # 0.5 * 1 + 0.5 * 0

def test_ensembler_median_wave(logger):
    wav1 = np.ones((2, 100))
    wav2 = np.zeros((2, 100))
    wav3 = np.ones((2, 100)) * 0.7
    ensembler = Ensembler(logger, algorithm="median_wave")
    result = ensembler.ensemble([wav1, wav2, wav3])
    assert np.allclose(result, 0.7)

def test_ensembler_max_wave(logger):
    wav1 = np.array([[1.0, -2.0], [3.0, -4.0]])
    wav2 = np.array([[0.5, -1.0], [4.0, -3.0]])
    ensembler = Ensembler(logger, algorithm="max_wave")
    result = ensembler.ensemble([wav1, wav2])
    # key=np.abs, so max of (1.0, 0.5) is 1.0, (-2.0, -1.0) is -2.0, (3.0, 4.0) is 4.0, (-4.0, -3.0) is -4.0
    expected = np.array([[1.0, -2.0], [4.0, -4.0]])
    assert np.allclose(result, expected)

def test_ensembler_min_wave(logger):
    wav1 = np.array([[1.0, -2.0], [3.0, -4.0]])
    wav2 = np.array([[0.5, -1.0], [4.0, -3.0]])
    ensembler = Ensembler(logger, algorithm="min_wave")
    result = ensembler.ensemble([wav1, wav2])
    # key=np.abs, so min of (1.0, 0.5) is 0.5, (-2.0, -1.0) is -1.0, (3.0, 4.0) is 3.0, (-4.0, -3.0) is -3.0
    expected = np.array([[0.5, -1.0], [3.0, -3.0]])
    assert np.allclose(result, expected)

def test_ensembler_avg_fft(logger):
    # FFT algorithms involve STFT/ISTFT which are harder to test with simple constants
    # but we can check if it returns a valid waveform of correct shape
    wav1 = np.random.rand(2, 1024)
    wav2 = np.random.rand(2, 1024)
    ensembler = Ensembler(logger, algorithm="avg_fft")
    result = ensembler.ensemble([wav1, wav2])
    assert result.shape == (2, 1024)

def test_ensembler_ensemble_wav_uvr(logger):
    # Linear Ensemble (least noisy chunk)
    wav1 = np.ones((2, 1000))
    wav2 = np.zeros((2, 1000))
    ensembler = Ensembler(logger, algorithm="ensemble_wav")
    # It splits into 240 chunks by default. Each chunk in wav2 is less noisy (all 0s)
    # so the result should be all 0s.
    result = ensembler.ensemble([wav1, wav2])
    assert np.allclose(result, 0.0)