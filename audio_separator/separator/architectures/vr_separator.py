"""Module for separating audio sources using VR architecture models."""

import os
import math

import torch
import librosa
import numpy as np
from tqdm import tqdm

# Check if we really need the rerun_mp3 function, remove if not
import audioread

from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.uvr_lib_v5.vr_network import nets
from audio_separator.separator.uvr_lib_v5.vr_network import nets_new
from audio_separator.separator.uvr_lib_v5.vr_network.model_param_init import ModelParameters


class VRSeparator(CommonSeparator):
    """
    VRSeparator is responsible for separating audio sources using VR models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def __init__(self, common_config, arch_config: dict):
        # Any configuration values which can be shared between architectures should be set already in CommonSeparator,
        # e.g. user-specified functionality choices (self.output_single_stem) or common model parameters (self.primary_stem_name)
        super().__init__(config=common_config)

        # Model data is basic overview metadata about the model, e.g. which stem is primary and whether it's a karaoke model
        # It's loaded in from model_data_new.json in Separator.load_model and there are JSON examples in that method
        # The instance variable self.model_data is passed through from Separator and set in CommonSeparator
        self.logger.debug(f"Model data: {self.model_data}")

        # Most of the VR models use the same number of output channels, but the VR 51 models have specific values set in model_data JSON
        self.model_capacity = 32, 128
        self.is_vr_51_model = False

        if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
            self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
            self.is_vr_51_model = True

        # Model params are additional technical parameter values from JSON files in separator/uvr_lib_v5/vr_network/modelparams/*.json,
        # with filenames referenced by the model_data["vr_model_param"] value
        package_root_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vr_params_json_dir = os.path.join(package_root_filepath, "uvr_lib_v5", "vr_network", "modelparams")
        vr_params_json_filename = f"{self.model_data['vr_model_param']}.json"
        vr_params_json_filepath = os.path.join(vr_params_json_dir, vr_params_json_filename)
        self.model_params = ModelParameters(vr_params_json_filepath)

        self.logger.debug(f"Model params: {self.model_params.param}")

        # Arch Config is the VR architecture specific user configuration options, which should all be configurable by the user
        # either by their Separator class instantiation or by passing in a CLI parameter.
        # While there are similarities between architectures for some of these (e.g. batch_size), they are deliberately configured
        # this way as they have architecture-specific default values.

        # This option performs Test-Time-Augmentation to improve the separation quality.
        # Note: Having this selected will increase the time it takes to complete a conversion
        self.enable_tta = arch_config.get("enable_tta", False)

        # This option can potentially identify leftover instrumental artifacts within the vocal outputs; may improve the separation of some songs.
        # Note: Selecting this option can adversely affect the conversion process, depending on the track. Because of this, it is only recommended as a last resort.
        self.enable_post_process = arch_config.get("enable_post_process", False)

        # post_process_threshold values = ('0.1', '0.2', '0.3')
        self.post_process_threshold = arch_config.get("post_process_threshold", 0.2)

        # Number of batches to be processed at a time.
        # - Higher values mean more RAM usage but slightly faster processing times.
        # - Lower values mean less RAM usage but slightly longer processing times.
        # - Batch size value has no effect on output quality.

        # Andrew note: for some reason, lower batch sizes seem to cause broken output for VR arch; need to investigate why
        self.batch_size = arch_config.get("batch_size", 1)

        # Select window size to balance quality and speed:
        # - 1024 - Quick but lesser quality.
        # - 512 - Medium speed and quality.
        # - 320 - Takes longer but may offer better quality.
        self.window_size = arch_config.get("window_size", 512)

        # The application will mirror the missing frequency range of the output.
        self.high_end_process = arch_config.get("high_end_process", False)
        self.input_high_end_h = None
        self.input_high_end = None

        # Adjust the intensity of primary stem extraction:
        # - Ranges from -100 - 100.
        # - Bigger values mean deeper extractions.
        # - Typically, it's set to 5 for vocals & instrumentals.
        # - Values beyond 5 might muddy the sound for non-vocal models.
        self.aggression = float(int(arch_config.get("aggression", 5)) / 100)

        self.aggressiveness = {"value": self.aggression, "split_bin": self.model_params.param["band"][1]["crop_stop"], "aggr_correction": self.model_params.param.get("aggr_correction")}

        self.model_samplerate = self.model_params.param["sr"]

        self.logger.debug(f"VR arch params: enable_tta={self.enable_tta}, enable_post_process={self.enable_post_process}, post_process_threshold={self.post_process_threshold}")
        self.logger.debug(f"VR arch params: batch_size={self.batch_size}, window_size={self.window_size}")
        self.logger.debug(f"VR arch params: high_end_process={self.high_end_process}, aggression={self.aggression}")
        self.logger.debug(f"VR arch params: is_vr_51_model={self.is_vr_51_model}, model_samplerate={self.model_samplerate}, model_capacity={self.model_capacity}")

        self.model_run = lambda *args, **kwargs: self.logger.error("Model run method is not initialised yet.")

        # wav_subtype will be set based on input audio bit depth in prepare_mix()
        # Removed hardcoded "PCM_16" to allow bit depth preservation

        self.logger.info("VR Separator initialisation complete")

    def separate(self, audio_file_path, custom_output_names=None):
        """
        Separates the audio file into primary and secondary sources based on the model's configuration.
        It processes the mix, demixes it into sources, normalizes the sources, and saves the output files.

        Args:
            audio_file_path (str): The path to the audio file to be processed.
            custom_output_names (dict, optional): Custom names for the output files. Defaults to None.

        Returns:
            list: A list of paths to the output files generated by the separation process.
        """
        self.primary_source = None
        self.secondary_source = None

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[ 0]

        # Detect input audio bit depth for output preservation
        try:
            import soundfile as sf
            info = sf.info(audio_file_path)
            self.input_audio_subtype = info.subtype
            self.logger.info(f"Input audio subtype: {self.input_audio_subtype}")
            
            # Map subtype to wav_subtype for soundfile and set input_bit_depth for pydub
            if "24" in self.input_audio_subtype:
                self.wav_subtype = "PCM_24"
                self.input_bit_depth = 24
                self.logger.info("Detected 24-bit input audio")
            elif "32" in self.input_audio_subtype:
                self.wav_subtype = "PCM_32"
                self.input_bit_depth = 32
                self.logger.info("Detected 32-bit input audio")
            else:
                self.wav_subtype = "PCM_16"
                self.input_bit_depth = 16
                self.logger.info("Detected 16-bit input audio")
        except Exception as e:
            self.logger.warning(f"Could not detect input audio bit depth: {e}. Defaulting to PCM_16")
            self.wav_subtype = "PCM_16"
            self.input_audio_subtype = None
            self.input_bit_depth = 16

        self.logger.debug(f"Starting separation for input audio file {self.audio_file_path}...")

        nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]  # default
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size))
        self.logger.debug(f"Model size determined: {model_size}, NN architecture size: {nn_arch_size}")

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.logger.debug("Using CascadedNet for VR 5.1 model...")
            self.model_run = nets_new.CascadedNet(self.model_params.param["bins"] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
            self.is_vr_51_model = True
        else:
            self.logger.debug("Determining model capacity...")
            self.model_run = nets.determine_model_capacity(self.model_params.param["bins"] * 2, nn_arch_size)

        self.model_run.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model_run.to(self.torch_device)
        self.logger.debug("Model loaded and moved to device.")

        y_spec, v_spec = self.inference_vr(self.loading_mix(), self.torch_device, self.aggressiveness)
        self.logger.debug("Inference completed.")

        # Sanitize y_spec and v_spec to replace NaN and infinite values
        y_spec = np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)
        v_spec = np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)

        self.logger.debug("Sanitization completed. Replaced NaN and infinite values in y_spec and v_spec.")

        # After inference_vr call
        self.logger.debug(f"Inference VR completed. y_spec shape: {y_spec.shape}, v_spec shape: {v_spec.shape}")
        self.logger.debug(f"y_spec stats - min: {np.min(y_spec)}, max: {np.max(y_spec)}, isnan: {np.isnan(y_spec).any()}, isinf: {np.isinf(y_spec).any()}")
        self.logger.debug(f"v_spec stats - min: {np.min(v_spec)}, max: {np.max(v_spec)}, isnan: {np.isnan(v_spec).any()}, isinf: {np.isinf(v_spec).any()}")

        # Not yet implemented from UVR features:
        #
        # if not self.is_vocal_split_model:
        #     self.cache_source((y_spec, v_spec))

        # if self.is_secondary_model_activated and self.secondary_model:
        #     self.logger.debug("Processing secondary model...")
        #     self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(
        #         self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem
        #     )

        # Initialize the list for output files
        output_files = []
        self.logger.debug("Processing output files...")

        # Note: logic similar to the following should probably be added to the other architectures
        # Check if output_single_stem is set to a value that would result in no output files
        if self.output_single_stem and (self.output_single_stem.lower() != self.primary_stem_name.lower() and self.output_single_stem.lower() != self.secondary_stem_name.lower()):
            # If so, reset output_single_stem to None to save both stems
            self.output_single_stem = None
            self.logger.warning(f"The output_single_stem setting '{self.output_single_stem}' does not match any of the output files: '{self.primary_stem_name}' and '{self.secondary_stem_name}'. For this model '{self.model_name}', the output_single_stem setting will be ignored and all output files will be saved.")

        # Save and process the primary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.logger.debug(f"Processing primary stem: {self.primary_stem_name}")
            if not isinstance(self.primary_source, np.ndarray):
                self.logger.debug(f"Preparing to convert spectrogram to waveform. Spec shape: {y_spec.shape}")

                self.primary_source = self.spec_to_wav(y_spec).T
                self.logger.debug("Converting primary source spectrogram to waveform.")
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                    self.logger.debug("Resampling primary source to 44100Hz.")

            self.primary_stem_output_path = self.get_stem_output_path(self.primary_stem_name, custom_output_names)

            self.logger.info(f"Saving {self.primary_stem_name} stem to {self.primary_stem_output_path}...")
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        # Save and process the secondary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.logger.debug(f"Processing secondary stem: {self.secondary_stem_name}")
            if not isinstance(self.secondary_source, np.ndarray):
                self.logger.debug(f"Preparing to convert spectrogram to waveform. Spec shape: {v_spec.shape}")

                self.secondary_source = self.spec_to_wav(v_spec).T
                self.logger.debug("Converting secondary source spectrogram to waveform.")
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                    self.logger.debug("Resampling secondary source to 44100Hz.")

            self.secondary_stem_output_path = self.get_stem_output_path(self.secondary_stem_name, custom_output_names)

            self.logger.info(f"Saving {self.secondary_stem_name} stem to {self.secondary_stem_output_path}...")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        # Not yet implemented from UVR features:
        # self.process_vocal_split_chain(secondary_sources)
        # self.logger.debug("Vocal split chain processed.")

        return output_files

    def loading_mix(self):
        X_wave, X_spec_s = {}, {}

        bands_n = len(self.model_params.param["band"])

        audio_file = spec_utils.write_array_to_mem(self.audio_file_path, subtype=self.wav_subtype)
        is_mp3 = audio_file.endswith(".mp3") if isinstance(audio_file, str) else False

        self.logger.debug(f"loading_mix iteraring through {bands_n} bands")
        for d in tqdm(range(bands_n, 0, -1)):
            bp = self.model_params.param["band"][d]

            wav_resolution = bp["res_type"]

            if self.torch_device_mps is not None:
                wav_resolution = "polyphase"

            if d == bands_n:  # high-end band
                X_wave[d], _ = librosa.load(audio_file, sr=bp["sr"], mono=False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params, band=d, is_v51_model=self.is_vr_51_model)

                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp["sr"])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.resample(X_wave[d + 1], orig_sr=self.model_params.param["band"][d + 1]["sr"], target_sr=bp["sr"], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.model_params.param["pre_filter_stop"] - self.model_params.param["pre_filter_start"])
                self.input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.model_params, is_v51_model=self.is_vr_51_model)

        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size

            self.logger.debug(f"inference_vr appending to X_dataset for each of {patches} patches")
            for i in tqdm(range(patches)):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start : start + self.window_size]
                X_dataset.append(X_mag_window)

            total_iterations = patches // self.batch_size if not self.enable_tta else (patches // self.batch_size) * 2
            self.logger.debug(f"inference_vr iterating through {total_iterations} batches, batch_size = {self.batch_size}")

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []

                for i in tqdm(range(0, patches, self.batch_size)):

                    X_batch = X_dataset[i : i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise ValueError(f"Window size error: h1_shape[3] must be greater than h2_shape[3]")
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise ValueError(f"Window size error: h1_shape[3] must be greater than h2_shape[3]")

                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in CommonSeparator.NON_ACCOM_STEMS:
                if stem == self.primary_stem_name:
                    is_non_accom_stem = True

            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.enable_post_process:
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

        if self.enable_tta:
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
        if self.high_end_process and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:
            input_high_end_ = spec_utils.mirroring("mirroring", spec, self.input_high_end, self.model_params)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, is_v51_model=self.is_vr_51_model)

        return wav


# Check if we really need the rerun_mp3 function, refactor or remove if not
def rerun_mp3(audio_file, sample_rate=44100):
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]
