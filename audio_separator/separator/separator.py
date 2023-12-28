import os
import warnings
import hashlib
import json
import logging
import warnings
import requests
import torch
import librosa
import numpy as np
import onnxruntime as ort
from pydub import AudioSegment
from audio_separator.separator import spec_utils
from audio_separator.separator.stft import STFT


class Separator:
    def __init__(
        self,
        audio_file_path,
        log_level=logging.DEBUG,
        log_formatter=None,
        model_name="UVR_MDXNET_KARA_2",
        model_file_dir="/tmp/audio-separator-models/",
        output_dir=None,
        primary_stem_path=None,
        secondary_stem_path=None,
        output_format="WAV",
        output_subtype=None,
        normalization_enabled=True,
        denoise_enabled=True,
        output_single_stem=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)

        self.logger.info(
            f"Separator instantiating with input file: {audio_file_path}, model_name: {model_name}, output_dir: {output_dir}, output_format: {output_format}"
        )

        self.model_name = model_name
        self.model_file_dir = model_file_dir
        self.output_dir = output_dir
        self.primary_stem_path = primary_stem_path
        self.secondary_stem_path = secondary_stem_path

        # Create the model directory if it does not exist
        os.makedirs(self.model_file_dir, exist_ok=True)

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.model_name = model_name
        self.model_url = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{self.model_name}.onnx"
        self.model_data_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"

        self.output_subtype = output_subtype
        self.output_format = output_format

        if self.output_format is None:
            self.output_format = "WAV"

        if self.output_subtype is None and output_format == "WAV":
            self.output_subtype = "PCM_16"

        self.normalization_enabled = normalization_enabled
        if self.normalization_enabled:
            self.logger.debug(f"Normalization enabled, waveform will be normalized to max amplitude of 1.0 to avoid clipping.")
        else:
            self.logger.debug(f"Normalization disabled, waveform will not be normalized.")

        self.denoise_enabled = denoise_enabled
        if self.denoise_enabled:
            self.logger.debug(f"Denoising enabled, model will be run twice to reduce noise in output audio.")
        else:
            self.logger.debug(
                f"Denoising disabled, model will only be run once. This is twice as fast, but may result in noisier output audio."
            )

        self.output_single_stem = output_single_stem
        if output_single_stem is not None:
            if output_single_stem.lower() not in {"instrumental", "vocals"}:
                raise Exception("output_single_stem must be either 'instrumental' or 'vocals'")
            self.logger.debug(f"Single stem output requested, only one output file ({output_single_stem}) will be written")

        self.chunks = 0
        self.margin = 44100
        self.adjust = 1
        self.dim_c = 4
        self.hop = 1024
        self.segment_size = 256

        self.primary_source = None
        self.secondary_source = None

        warnings.filterwarnings("ignore")
        self.cpu = torch.device("cpu")

        # Prepare for hardware-accelerated inference by validating both Torch and ONNX Runtime support either CUDA or CoreML
        self.logger.debug(f"Torch version: {str(torch.__version__)}")
        ort_device = ort.get_device()
        ort_providers = ort.get_available_providers()
        hardware_acceleration_enabled = False

        if torch.cuda.is_available():
            self.logger.info("CUDA is available in Torch, setting Torch device to CUDA")
            self.device = torch.device("cuda")

            if ort_device == "GPU" and "CUDAExecutionProvider" in ort_providers:
                self.logger.info("ONNXruntime has CUDAExecutionProvider available, enabling acceleration")
                self.onnx_execution_provider = ["CUDAExecutionProvider"]
                hardware_acceleration_enabled = True
            else:
                self.logger.warning("CUDAExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")
                self.logger.warning("If you expect CUDA to work with your GPU, try pip install --force-reinstall onnxruntime-gpu")
        else:
            self.logger.debug("CUDA not available in Torch installation. If you expect GPU/CUDA support to work, please see README")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.logger.info("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")

            # TODO: Change this to use MPS once FFTs are supported, see https://github.com/pytorch/pytorch/issues/78044
            # self.device = torch.device("mps")

            self.logger.warning("Torch MPS backend does not yet support FFT operations, Torch will still use CPU!")
            self.logger.warning("To track progress towards Apple Silicon acceleration, see https://github.com/pytorch/pytorch/issues/78044")
            self.device = torch.device("cpu")

            if "CoreMLExecutionProvider" in ort_providers:
                self.logger.info("ONNXruntime has CoreMLExecutionProvider available, enabling acceleration")
                self.onnx_execution_provider = ["CoreMLExecutionProvider"]
                hardware_acceleration_enabled = True
            else:
                self.logger.warning("CoreMLExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")
                self.logger.warning("If you expect MPS/CoreML to work with your Mac, try pip install --force-reinstall onnxruntime-silicon")
        else:
            self.logger.debug("Apple Silicon MPS/CoreML not available in Torch installation. If you expect this to work, please see README")

        if not hardware_acceleration_enabled:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
            self.device = torch.device("cpu")
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def get_model_hash(self, model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file(self, url, output_path):
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            self.logger.error(f"Failed to download file from {url}")

    def separate(self):
        self.logger.debug("Starting separation process...")
        model_path = os.path.join(self.model_file_dir, f"{self.model_name}.onnx")
        self.logger.debug(f"Model path set to {model_path}")

        if not os.path.isfile(model_path):
            self.logger.debug(f"Model not found at path {model_path}, downloading...")
            self.download_file(self.model_url, model_path)

        self.logger.debug("Reading model settings...")
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(f"Model {model_path} has hash {model_hash}")

        model_data_path = os.path.join(self.model_file_dir, "model_data.json")
        self.logger.debug(f"Model data path set to {model_data_path}")

        if not os.path.isfile(model_data_path):
            self.logger.debug(f"Model data not found at path {model_data_path}, downloading...")
            self.download_file(self.model_data_url, model_data_path)

        self.logger.debug("Loading model data...")
        model_data_object = json.load(open(model_data_path))
        model_data = model_data_object[model_hash]
        self.logger.debug(f"Model data loaded: {model_data}")

        self.compensate, self.dim_f, self.dim_t, self.n_fft, self.primary_stem = (
            model_data["compensate"],
            model_data["mdx_dim_f_set"],
            2 ** model_data["mdx_dim_t_set"],
            model_data["mdx_n_fft_scale_set"],
            model_data["primary_stem"],
        )
        self.secondary_stem = "Vocals" if self.primary_stem == "Instrumental" else "Instrumental"
        self.logger.debug(
            f"Model settings initialized: compensate={self.compensate}, primary_stem={self.primary_stem}, secondary_stem={self.secondary_stem}, dim_f={self.dim_f}, dim_t={self.dim_t}, n_fft={self.n_fft}"
        )

        self.logger.debug("Loading ONNX model for inference...")
        ort_ = ort.InferenceSession(model_path, providers=self.onnx_execution_provider)
        self.model_run = lambda spek: ort_.run(None, {"input": spek.cpu().numpy()})[0]
        self.logger.debug("Model loaded successfully.")

        self.initialize_model_settings()
        self.logger.info("Model settings initialized, starting inference...")

        mdx_net_cut = True  # Hardcoded for now, can be parameterized if needed
        self.logger.debug(f"Preparing mix with chunks={self.chunks}, margin={self.margin}, mdx_net_cut={mdx_net_cut}")
        mix, raw_mix, samplerate = self.prepare_mix(self.audio_file_path, self.chunks, self.margin, mdx_net_cut=mdx_net_cut)

        self.logger.info("Starting demixing process...")
        source = self.demix_base(mix)[0]
        self.logger.debug("Demixing completed.")

        output_files = []
        self.logger.debug("Processing output files...")

        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug("Normalizing primary source...")
            self.primary_source = spec_utils.normalize(self.logger, source, self.normalization_enabled).T

        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug("Normalizing secondary source...")
            raw_mix = self.demix_base(raw_mix, is_match_mix=True)[0] if mdx_net_cut else raw_mix
            self.secondary_source, raw_mix = spec_utils.normalize_two_stem(
                self.logger, source * self.compensate, raw_mix, self.normalization_enabled
            )
            self.secondary_source = -self.secondary_source.T + raw_mix.T

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem.lower():
            self.logger.info(f"Saving {self.primary_stem} stem...")
            if not self.primary_stem_path:
                self.primary_stem_path = os.path.join(
                    f"{self.audio_file_base}_({self.primary_stem})_{self.model_name}.{self.output_format.lower()}"
                )
            self.write_audio(self.primary_stem_path, self.primary_source, samplerate)
            output_files.append(self.primary_stem_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem.lower():
            self.logger.info(f"Saving {self.secondary_stem} stem...")
            if not self.secondary_stem_path:
                self.secondary_stem_path = os.path.join(
                    f"{self.audio_file_base}_({self.secondary_stem})_{self.model_name}.{self.output_format.lower()}"
                )
            self.write_audio(self.secondary_stem_path, self.secondary_source, samplerate)
            output_files.append(self.secondary_stem_path)

        if hasattr(torch, "cuda"):
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()

        self.logger.debug("Separation process completed.")
        return output_files

    def write_audio(self, stem_path, stem_source, samplerate):
        self.logger.debug(f"Entering write_audio with stem_path: {stem_path}, samplerate: {samplerate}")

        # Check if the numpy array is empty or contains very low values
        if np.max(np.abs(stem_source)) < 1e-6:
            self.logger.warning("Warning: stem_source array is near-silent or empty.")
            return

        # If output_dir is specified, create it and join it with stem_path
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)

        self.logger.debug(f"Audio data shape before processing: {stem_source.shape}")
        self.logger.debug(f"Data type before conversion: {stem_source.dtype}")

        # Ensure the audio data is in the correct format (e.g., int16)
        if stem_source.dtype != np.int16:
            stem_source = (stem_source * 32767).astype(np.int16)
            self.logger.debug("Converted stem_source to int16.")

        # Correctly interleave stereo channels
        stem_source_interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)
        stem_source_interleaved[0::2] = stem_source[:, 0]  # Left channel
        stem_source_interleaved[1::2] = stem_source[:, 1]  # Right channel

        self.logger.debug(f"Interleaved audio data shape: {stem_source_interleaved.shape}")

        # Create a pydub AudioSegment
        try:
            audio_segment = AudioSegment(
                stem_source_interleaved.tobytes(), frame_rate=samplerate, sample_width=stem_source.dtype.itemsize, channels=2
            )
            self.logger.debug("Created AudioSegment successfully.")
        except Exception as e:
            self.logger.error(f"Error creating AudioSegment: {e}")
            return

        # Determine file format based on the file extension
        file_format = stem_path.lower().split(".")[-1]

        # For M4A files, specify 'mp4' as the container format
        if file_format == "m4a":
            file_format = "mp4"
        elif file_format == "mka":
            file_format = "matroska"

        # Export using the determined format
        try:
            audio_segment.export(stem_path, format=file_format)
            self.logger.debug(f"Exported audio file successfully to {stem_path}")
        except Exception as e:
            self.logger.error(f"Error exporting audio file: {e}")

    # This function sets up the necessary parameters for the model, like the number of frequency bins (n_bins), the trimming size (trim),
    # the size of each audio chunk (chunk_size), and the window function for spectral transformations (window).
    # It ensures that the model is configured with the correct settings for processing the audio data.
    def initialize_model_settings(self):
        self.logger.debug("Initializing model settings...")

        # n_bins is half the FFT size plus one (self.n_fft // 2 + 1).
        self.n_bins = self.n_fft // 2 + 1

        # trim is half the FFT size (self.n_fft // 2).
        self.trim = self.n_fft // 2

        # chunk_size is the hop size times the segment size (hard-coded in init; in UVR it's selectable in the UI) minus one
        self.chunk_size = self.hop * (self.segment_size - 1)

        # gen_size is the chunk size minus twice the trim size
        self.gen_size = self.chunk_size - 2 * self.trim

        self.stft = STFT(self.logger, self.n_fft, self.hop, self.dim_f, self.device)
        self.logger.debug(
            f"Model settings and STFT initialized: n_fft={self.n_fft} hop={self.hop} dim_f={self.dim_f} n_bins={self.n_bins}, trim={self.trim}, chunk_size={self.chunk_size}, gen_size={self.gen_size}"
        )

    # After prepare_mix segments the audio, initialize_mix further processes each segment.
    # It ensures each audio segment is in the correct format for the model, applies necessary padding,
    # and converts the segments into tensors for processing with the model.
    # This step is essential for preparing the audio data in a format that the neural network can process.
    def initialize_mix(self, mix, is_ckpt=False):
        self.logger.debug(f"Initializing mix with is_ckpt={is_ckpt}. Initial mix shape: {mix.shape}")

        if mix.shape[0] != 2:
            error_message = f"Expected a 2-channel audio signal, but got {mix.shape[0]} channels"
            self.logger.error(error_message)
            raise ValueError(error_message)

        if is_ckpt:
            self.logger.debug("Processing in checkpoint mode...")
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            self.logger.debug(f"Padding calculated: {pad}")
            mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)
            num_chunks = mixture.shape[-1] // self.gen_size
            self.logger.debug(f"Mixture shape after padding: {mixture.shape}, Number of chunks: {num_chunks}")
            mix_waves = [mixture[:, i * self.gen_size : i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            self.logger.debug("Processing in non-checkpoint mode...")
            mix_waves = []
            n_sample = mix.shape[1]
            pad = self.gen_size - n_sample % self.gen_size
            self.logger.debug(f"Number of samples: {n_sample}, Padding calculated: {pad}")
            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
            self.logger.debug(f"Shape of mix after padding: {mix_p.shape}")

            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + self.chunk_size])
                mix_waves.append(waves)
                self.logger.debug(f"Processed chunk {len(mix_waves)}: Start {i}, End {i + self.chunk_size}")
                i += self.gen_size

        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        self.logger.debug(f"Converted mix_waves to tensor. Tensor shape: {mix_waves_tensor.shape}")

        return mix_waves_tensor, pad

    # This is the core function where the actual separation of vocals and instrumentals happens.
    # It iterates over each audio chunk, applies the model to each chunk (via run_model), and then concatenates the results.
    # This function is where the heavy lifting of source separation occurs.
    def demix_base(self, mix, is_ckpt=False, is_match_mix=False):
        self.logger.debug(f"Starting demixing base method. is_ckpt={is_ckpt}, is_match_mix={is_match_mix}")
        chunked_sources = []

        for slice in mix:
            self.logger.debug(f"Processing slice {slice}")
            sources = []
            tar_waves_ = []
            mix_p = mix[slice]
            mix_waves, pad = self.initialize_mix(mix_p, is_ckpt=is_ckpt)
            self.logger.debug(f"mix_waves shape: {mix_waves.shape}, pad: {pad}")

            mix_waves = mix_waves.split(1)
            pad = mix_p.shape[-1] if is_ckpt else -pad

            with torch.no_grad():
                for mix_wave in mix_waves:
                    tar_waves = self.run_model(mix_wave, is_ckpt=is_ckpt, is_match_mix=is_match_mix)
                    tar_waves_.append(tar_waves)
                tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim : -self.trim] if is_ckpt else tar_waves_
                tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :pad]
                self.logger.debug(f"Concatenated tar_waves shape: {tar_waves.shape}")

                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
                sources.append(tar_waves[:, start:end] * (1 / self.adjust))
            chunked_sources.append(sources)
        sources = np.concatenate(chunked_sources, axis=-1)
        self.logger.debug(f"Final concatenated sources shape: {sources.shape}")

        return sources

    # This function is called by demix_base for each audio chunk.
    # It applies a Short-Time Fourier Transform (STFT) to the chunk, processes it through the neural network model,
    # and then applies an inverse STFT to convert it back to the time domain.
    # This function is where the model infers the separation of vocals and instrumentals from the mixed audio.
    def run_model(self, mix, is_ckpt=False, is_match_mix=False):
        self.logger.debug(f"Running model on mix_wave with is_ckpt={is_ckpt}, is_match_mix={is_match_mix}")
        spek = self.stft(mix.to(self.device)) * self.adjust

        spek[:, :, :3, :] *= 0
        self.logger.debug("Zeroed the first 3 bins of the spectrum")

        if is_match_mix:
            self.logger.debug("Running model in match mix mode...")
            spec_pred = spek.cpu().numpy()
        else:
            self.logger.debug("Running model in normal mode...")
            spec_pred = -self.model_run(-spek) * 0.5 + self.model_run(spek) * 0.5 if self.denoise_enabled else self.model_run(spek)

        if is_ckpt:
            return self.stft.inverse(spec_pred).cpu().detach().numpy()
        else:
            return (
                self.stft.inverse(torch.tensor(spec_pred).to(self.device))
                .to(self.cpu)[:, :, self.trim : -self.trim]
                .transpose(0, 1)
                .reshape(2, -1)
                .numpy()
            )

    # This function handles the initial processing of the audio file. It involves loading the audio file (or array),
    # ensuring it's in stereo format, and then segmenting it into manageable chunks based on the specified chunk size and margin.
    # This segmentation is crucial for efficient processing of the audio, especially for longer tracks.
    def prepare_mix(self, mix, chunk_set, margin_set, mdx_net_cut=False, is_missing_mix=False):
        self.logger.debug(f"Starting to prepare mix. Chunk set: {chunk_set}, Margin set: {margin_set}, MDX Net Cut: {mdx_net_cut}")

        samplerate = 44100
        self.logger.debug(f"Default samplerate set to {samplerate}")

        # Load mix if it's a file path, or transpose if it's an array
        if not isinstance(mix, np.ndarray):
            self.logger.debug(f"Loading mix from file: {mix}")
            mix, samplerate = librosa.load(mix, mono=False, sr=samplerate)
        else:
            self.logger.debug("Transposing the given mix array.")
            mix = mix.T

        # Ensure the mix is stereo
        if mix.ndim == 1:
            self.logger.debug("Mix is mono. Converting to stereo.")
            mix = np.asfortranarray([mix, mix])

        def get_segmented_mix(chunk_set=chunk_set):
            self.logger.debug(f"Segmenting mix. Chunk size: {chunk_set}, Margin: {margin_set}")

            segmented_mix = {}
            samples = mix.shape[-1]
            margin = margin_set
            chunk_size = chunk_set * samplerate

            self.logger.debug(f"Calculated chunk size: {chunk_size}, Total samples: {samples}")

            if margin > chunk_size:
                self.logger.debug(f"Margin ({margin}) is greater than chunk size. Setting margin to chunk size.")
                margin = chunk_size

            if chunk_set == 0 or samples < chunk_size:
                self.logger.debug("Chunk set is zero or samples less than chunk size. Setting chunk size to total samples.")
                chunk_size = samples

            counter = -1
            for skip in range(0, samples, chunk_size):
                counter += 1
                s_margin = 0 if counter == 0 else margin
                end = min(skip + chunk_size + margin, samples)
                start = skip - s_margin

                self.logger.debug(f"Processing chunk {counter}. Start: {start}, End: {end}")

                segmented_mix[skip] = mix[:, start:end].copy()
                if end == samples:
                    self.logger.debug("Reached end of samples.")
                    break

            return segmented_mix

        segmented_mix = get_segmented_mix()
        raw_mix = get_segmented_mix(chunk_set=0) if mdx_net_cut else mix

        self.logger.debug("Mix preparation completed.")

        return segmented_mix, raw_mix, samplerate
