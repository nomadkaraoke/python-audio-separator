import numpy as np
import librosa
from audio_separator.separator.uvr_lib_v5 import spec_utils


class Ensembler:
    def __init__(self, logger, algorithm="avg_wave", weights=None):
        self.logger = logger
        self.algorithm = algorithm
        self.weights = weights

    def ensemble(self, waveforms):
        """
        Ensemble multiple waveforms using the selected algorithm.
        :param waveforms: List of waveforms, each of shape (channels, length)
        :return: Ensembled waveform of shape (channels, length)
        """
        if not waveforms:
            return None
        if len(waveforms) == 1:
            return waveforms[0]

        # Ensure all waveforms have the same length by padding with zeros
        max_length = max(w.shape[1] for w in waveforms)
        waveforms = [np.pad(w, ((0, 0), (0, max_length - w.shape[1]))) if w.shape[1] < max_length else w for w in waveforms]

        if self.weights is None:
            self.weights = np.ones(len(waveforms))
        else:
            self.weights = np.array(self.weights)
            if len(self.weights) != len(waveforms):
                self.logger.warning(f"Number of weights ({len(self.weights)}) does not match number of waveforms ({len(waveforms)}). Using equal weights.")
                self.weights = np.ones(len(waveforms))

        self.logger.debug(f"Ensembling {len(waveforms)} waveforms using algorithm {self.algorithm}")

        if self.algorithm == "avg_wave":
            ensembled = np.zeros_like(waveforms[0])
            for w, weight in zip(waveforms, self.weights):
                ensembled += w * weight
            return ensembled / np.sum(self.weights)
        elif self.algorithm == "median_wave":
            return np.median(waveforms, axis=0)
        elif self.algorithm == "min_wave":
            return self._lambda_min(np.array(waveforms), axis=0, key=np.abs)
        elif self.algorithm == "max_wave":
            return self._lambda_max(np.array(waveforms), axis=0, key=np.abs)
        elif self.algorithm in ["avg_fft", "median_fft", "min_fft", "max_fft"]:
            return self._ensemble_fft(waveforms)
        elif self.algorithm == "uvr_max_spec":
            return self._ensemble_uvr(waveforms, spec_utils.MAX_SPEC)
        elif self.algorithm == "uvr_min_spec":
            return self._ensemble_uvr(waveforms, spec_utils.MIN_SPEC)
        elif self.algorithm == "ensemble_wav":
            return spec_utils.ensemble_wav(waveforms)
        else:
            raise ValueError(f"Unknown ensemble algorithm: {self.algorithm}")

    def _lambda_max(self, arr, axis=None, key=None, keepdims=False):
        idxs = np.argmax(key(arr), axis)
        if axis is not None:
            idxs = np.expand_dims(idxs, axis)
            result = np.take_along_axis(arr, idxs, axis)
            if not keepdims:
                result = np.squeeze(result, axis=axis)
            return result
        else:
            return arr.flatten()[idxs]

    def _lambda_min(self, arr, axis=None, key=None, keepdims=False):
        idxs = np.argmin(key(arr), axis)
        if axis is not None:
            idxs = np.expand_dims(idxs, axis)
            result = np.take_along_axis(arr, idxs, axis)
            if not keepdims:
                result = np.squeeze(result, axis=axis)
            return result
        else:
            return arr.flatten()[idxs]

    def _stft(self, wave, nfft=2048, hl=1024):
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])
        spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
        spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
        spec = np.asfortranarray([spec_left, spec_right])
        return spec

    def _istft(self, spec, hl=1024, length=None):
        spec_left = np.asfortranarray(spec[0])
        spec_right = np.asfortranarray(spec[1])
        wave_left = librosa.istft(spec_left, hop_length=hl, length=length)
        wave_right = librosa.istft(spec_right, hop_length=hl, length=length)
        wave = np.asfortranarray([wave_left, wave_right])
        return wave

    def _ensemble_fft(self, waveforms):
        final_length = waveforms[0].shape[-1]
        specs = [self._stft(w) for w in waveforms]
        specs = np.array(specs)

        if self.algorithm == "avg_fft":
            ense_spec = np.zeros_like(specs[0])
            for s, weight in zip(specs, self.weights):
                ense_spec += s * weight
            ense_spec /= np.sum(self.weights)
        elif self.algorithm == "median_fft":
            # For complex numbers, we take median of real and imag parts separately to be safe
            real_median = np.median(np.real(specs), axis=0)
            imag_median = np.median(np.imag(specs), axis=0)
            ense_spec = real_median + 1j * imag_median
        elif self.algorithm == "min_fft":
            ense_spec = self._lambda_min(specs, axis=0, key=np.abs)
        elif self.algorithm == "max_fft":
            ense_spec = self._lambda_max(specs, axis=0, key=np.abs)

        return self._istft(ense_spec, length=final_length)

    def _ensemble_uvr(self, waveforms, uvr_algorithm):
        specs = [spec_utils.wave_to_spectrogram_no_mp(w) for w in waveforms]
        ense_spec = spec_utils.ensembling(uvr_algorithm, specs)
        return spec_utils.spectrogram_to_wave_no_mp(ense_spec)