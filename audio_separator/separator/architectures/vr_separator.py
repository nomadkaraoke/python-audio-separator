"""Module for separating audio sources using VR architecture models."""

import os
import torch
import librosa
import onnxruntime as ort
import numpy as np
import onnx2torch
from audio_separator.separator import spec_utils
from audio_separator.separator.stft import STFT
from audio_separator.separator.common_separator import CommonSeparator


class VRSeparator(CommonSeparator):
    """
    VRSeparator is responsible for separating audio sources using VR models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        self.hop_length = arch_config.get("hop_length")
        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap")
        self.batch_size = arch_config.get("batch_size")

        self.logger.debug(f"Model params: primary_stem={self.primary_stem_name}, secondary_stem={self.secondary_stem_name}")
        self.logger.debug(f"Model params: batch_size={self.batch_size}, compensate={self.compensate}, segment_size={self.segment_size}, dim_f={self.dim_f}, dim_t={self.dim_t}")
        self.logger.debug(f"Model params: n_fft={self.n_fft}, hop={self.hop_length}")

        # Loading the model for inference
        self.logger.debug("Loading ONNX model for inference...")
        if self.segment_size == self.dim_t:
            ort_ = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider)
            self.model_run = lambda spek: ort_.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Model loaded successfully using ONNXruntime inferencing session.")
        else:
            self.model_run = onnx2torch.convert(self.model_path)
            self.model_run.to(self.torch_device).eval()
            self.logger.warning("Model converted from onnx to pytorch due to segment size not matching dim_t, processing may be slower.")

        self.n_bins = None
        self.trim = None
        self.chunk_size = None
        self.gen_size = None
        self.stft = None

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.secondary_source_map = None
        self.primary_source_map = None






    def seperate(self):
        self.logger.debug("Starting separation process in SeperateVR...")
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            self.logger.debug("Using cached primary sources...")
            y_spec, v_spec = self.primary_sources
            self.load_cached_sources()
        else:
            self.logger.debug("Starting inference...")
            self.start_inference_console_write()

            device = self.device
            self.logger.debug(f"Device set to: {device}")

            nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]  # default
            vr_5_1_models = [56817, 218409]
            model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
            nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size))
            self.logger.debug(f"Model size determined: {model_size}, NN architecture size: {nn_arch_size}")

            if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
                self.logger.debug("Using CascadedNet for VR 5.1 model...")
                self.model_run = nets_new.CascadedNet(self.mp.param["bins"] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
                self.is_vr_51_model = True
            else:
                self.logger.debug("Determining model capacity...")
                self.model_run = nets.determine_model_capacity(self.mp.param["bins"] * 2, nn_arch_size)

            self.model_run.load_state_dict(torch.load(self.model_path, map_location=cpu))
            self.model_run.to(device)
            self.logger.debug("Model loaded and moved to device.")

            self.running_inference_console_write()

            y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
            self.logger.debug("Inference completed.")
            if not self.is_vocal_split_model:
                self.cache_source((y_spec, v_spec))
            self.write_to_console(DONE, base_text="")

        if self.is_secondary_model_activated and self.secondary_model:
            self.logger.debug("Processing secondary model...")
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(
                self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem
            )

        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(self.export_path, f"{self.audio_file_base}_({self.primary_stem}).wav")
            self.logger.debug(f"Processing primary stem: {self.primary_stem}")
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = self.spec_to_wav(y_spec).T
                self.logger.debug("Converting primary source spectrogram to waveform.")
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                    self.logger.debug("Resampling primary source to 44100Hz.")

            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, 44100)
            self.logger.debug("Primary stem processed.")

        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f"{self.audio_file_base}_({self.secondary_stem}).wav")
            self.logger.debug(f"Processing secondary stem: {self.secondary_stem}")
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec).T
                self.logger.debug("Converting secondary source spectrogram to waveform.")
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                    self.logger.debug("Resampling secondary source to 44100Hz.")

            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, 44100)
            self.logger.debug("Secondary stem processed.")

        clear_gpu_cache()
        self.logger.debug("GPU cache cleared.")
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}

        self.process_vocal_split_chain(secondary_sources)
        self.logger.debug("Vocal split chain processed.")

        if self.is_secondary_model:
            self.logger.debug("Returning secondary sources...")
            return secondary_sources

    def loading_mix(self):
        X_wave, X_spec_s = {}, {}

        bands_n = len(self.mp.param["band"])

        audio_file = spec_utils.write_array_to_mem(self.audio_file, subtype=self.wav_type_set)
        is_mp3 = audio_file.endswith(".mp3") if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]

            if OPERATING_SYSTEM == "Darwin":
                wav_resolution = "polyphase" if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp["res_type"]
            else:
                wav_resolution = bp["res_type"]

            if d == bands_n:  # high-end band
                X_wave[d], _ = librosa.load(audio_file, bp["sr"], False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.mp, band=d, is_v51_model=self.is_vr_51_model)

                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp["sr"])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.resample(X_wave[d + 1], self.mp.param["band"][d + 1]["sr"], bp["sr"], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.mp, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process != "none":
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"])
                self.input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp, is_v51_model=self.is_vr_51_model)

        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches // self.batch_size if not self.is_tta else (patches // self.batch_size) * 2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start : start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    self.progress_value += 1
                    if self.progress_value >= total_iterations:
                        self.progress_value = total_iterations
                    self.set_progress_bar(0.1, 0.8 / total_iterations * self.progress_value)
                    X_batch = X_dataset[i : i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])

                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True

            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.0j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)

            return y_spec, v_spec

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)

        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2 :]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process.startswith("mirroring") and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, is_v51_model=self.is_vr_51_model)

        return wav
