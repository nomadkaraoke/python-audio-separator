import os
import gc
import platform
import hashlib
import json
import time
import logging
import warnings
import requests
import torch
import librosa
import numpy as np
import onnxruntime as ort
from importlib import metadata
from onnx2torch import convert
from pydub import AudioSegment
from audio_separator.separator import spec_utils
from audio_separator.separator.stft import STFT


class Separator:
    def __init__(
        self,
        log_level=logging.DEBUG,
        log_formatter=None,
        model_file_dir="/tmp/audio-separator-models/",
        output_dir=None,
        primary_stem_path=None,
        secondary_stem_path=None,
        output_format="WAV",
        output_subtype=None,
        normalization_threshold=0.9,
        denoise_enabled=False,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        hop_length=1024,
        segment_size=256,
        overlap=0.25,
        batch_size=1,
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

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        package_version = self.get_package_distribution("audio-separator").version

        self.logger.info(f"Separator version {package_version} instantiating with output_dir: {output_dir}, output_format: {output_format}")

        self.model_file_dir = model_file_dir
        self.output_dir = output_dir
        self.primary_stem_path = primary_stem_path
        self.secondary_stem_path = secondary_stem_path

        # Create the model directory if it does not exist
        os.makedirs(self.model_file_dir, exist_ok=True)

        self.output_subtype = output_subtype
        self.output_format = output_format

        if self.output_format is None:
            self.output_format = "WAV"

        if self.output_subtype is None and output_format == "WAV":
            self.output_subtype = "PCM_16"

        self.normalization_threshold = normalization_threshold
        self.logger.debug(
            f"Normalization threshold set to {normalization_threshold}, waveform will lowered to this max amplitude to avoid clipping."
        )

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

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec:
            self.logger.debug(
                f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality, but is slightly slower."
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.segment_size = segment_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.logger.debug(
            f"Separation settings set: sample_rate={self.sample_rate}, hop_length={self.hop_length}, segment_size={self.segment_size}, overlap={self.overlap}, batch_size={self.batch_size}"
        )

        self.setup_inferencing_device()

    def setup_inferencing_device(self):
        self.logger.info(f"Checking hardware specifics to configure acceleration")

        os_name = platform.system()
        os_version = platform.version()
        self.logger.info(f"Operating System: {os_name} {os_version}")

        system_info = platform.uname()
        self.logger.info(
            f"System: {system_info.system} Node: {system_info.node} Release: {system_info.release} Machine: {system_info.machine} Proc: {system_info.processor}"
        )

        python_version = platform.python_version()
        self.logger.info(f"Python Version: {python_version}")

        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        if onnxruntime_gpu_package is not None:
            self.logger.info(f"ONNX Runtime GPU package installed with version: {onnxruntime_gpu_package.version}")

        onnxruntime_silicon_package = self.get_package_distribution("onnxruntime-silicon")
        if onnxruntime_silicon_package is not None:
            self.logger.info(f"ONNX Runtime Silicon package installed with version: {onnxruntime_silicon_package.version}")

        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")
        if onnxruntime_cpu_package is not None:
            self.logger.info(f"ONNX Runtime CPU package installed with version: {onnxruntime_cpu_package.version}")

        torch_package = self.get_package_distribution("torch")
        if torch_package is not None:
            self.logger.info(f"Torch package installed with version: {torch_package.version}")

        torchvision_package = self.get_package_distribution("torchvision")
        if torchvision_package is not None:
            self.logger.info(f"Torchvision package installed with version: {torchvision_package.version}")

        torchaudio_package = self.get_package_distribution("torchaudio")
        if torchaudio_package is not None:
            self.logger.info(f"Torchaudio package installed with version: {torchaudio_package.version}")

        ort_device = ort.get_device()
        ort_providers = ort.get_available_providers()

        self.cpu = torch.device("cpu")
        hardware_acceleration_enabled = False

        # Prepare for hardware-accelerated inference by validating both Torch and ONNX Runtime support either CUDA or CoreML
        if torch.cuda.is_available():
            self.logger.info("CUDA is available in Torch, setting Torch device to CUDA")
            self.device = torch.device("cuda")

            if onnxruntime_gpu_package is not None and ort_device == "GPU" and "CUDAExecutionProvider" in ort_providers:
                self.logger.info("ONNXruntime has CUDAExecutionProvider available, enabling acceleration")
                self.onnx_execution_provider = ["CUDAExecutionProvider"]
                hardware_acceleration_enabled = True
            else:
                self.logger.warning("CUDAExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")
                self.logger.warning("If you expect CUDA to work with your GPU, try pip install --force-reinstall onnxruntime-gpu")
        else:
            self.logger.debug("CUDA not available in Torch installation. If you expect GPU/CUDA support to work, please see README")

        if onnxruntime_silicon_package is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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

    def get_package_distribution(self, package_name):
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(f"Python package: {package_name} not installed")
            return None

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

    def final_process(self, stem_path, source, stem_name, sample_rate):
        self.logger.debug(f"Finalizing {stem_name} stem processing and writing audio...")
        self.write_audio(stem_path, source, sample_rate, stem_name=stem_name)

        return {stem_name: source}

    def clear_gpu_cache(self):
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if self.device == torch.device("mps"):
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if self.device == torch.device("cuda"):
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()

    def load_model(self, model_name="UVR-MDX-NET-Inst_HQ_3"):
        self.logger.info(f"Loading model {model_name}...")

        self.load_model_start_time = time.perf_counter()

        self.model_name = model_name
        self.model_url = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{self.model_name}.onnx"
        self.model_data_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"

        # Setting up the model path
        model_path = os.path.join(self.model_file_dir, f"{self.model_name}.onnx")
        self.logger.debug(f"Model path set to {model_path}")

        # Check if model file exists, if not, download it
        if not os.path.isfile(model_path):
            self.logger.debug(f"Model not found at path {model_path}, downloading...")
            self.download_file(self.model_url, model_path)

        # Reading model settings from the downloaded model
        self.logger.debug("Reading model settings...")
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(f"Model {model_path} has hash {model_hash}")

        # Setting up the path for model data and checking its existence
        model_data_path = os.path.join(self.model_file_dir, "model_data.json")
        self.logger.debug(f"Model data path set to {model_data_path}")
        if not os.path.isfile(model_data_path):
            self.logger.debug(f"Model data not found at path {model_data_path}, downloading...")
            self.download_file(self.model_data_url, model_data_path)

        # Loading model data
        self.logger.debug("Loading model data...")
        model_data_object = json.load(open(model_data_path))
        model_data = model_data_object[model_hash]
        self.logger.debug(f"Model data loaded: {model_data}")

        # Initializing model parameters
        self.compensate, self.dim_f, self.dim_t, self.n_fft, self.model_primary_stem = (
            model_data["compensate"],
            model_data["mdx_dim_f_set"],
            2 ** model_data["mdx_dim_t_set"],
            model_data["mdx_n_fft_scale_set"],
            model_data["primary_stem"],
        )
        self.model_secondary_stem = "Vocals" if self.model_primary_stem == "Instrumental" else "Instrumental"

        # In UVR, these variables are set but either aren't useful or are better handled in audio-separator.
        # Leaving these comments explaining to help myself or future developers understand why these aren't in audio-separator.

        # "chunks" is not actually used for anything in UVR...
        # self.chunks = 0

        # "adjust" is hard-coded to 1 in UVR, and only used as a multiplier in run_model, so it does nothing.
        # self.adjust = 1

        # "hop" is hard-coded to 1024 in UVR. We have a "hop_length" parameter instead
        # self.hop = 1024

        # "margin" maps to sample rate and is set from the GUI in UVR (default: 44100). We have a "sample_rate" parameter instead.
        # self.margin = 44100

        # "dim_c" is hard-coded to 4 in UVR, seems to be a parameter for the number of channels, and is only used for checkpoint models.
        # We haven't implemented support for the checkpoint models here, so we're not using it.
        # self.dim_c = 4

        self.logger.debug(f"Model params: primary_stem={self.model_primary_stem}, secondary_stem={self.model_secondary_stem}")
        self.logger.debug(
            f"Model params: batch_size={self.batch_size}, compensate={self.compensate}, segment_size={self.segment_size}, dim_f={self.dim_f}, dim_t={self.dim_t}"
        )
        self.logger.debug(f"Model params: n_fft={self.n_fft}, hop={self.hop_length}")

        # Loading the model for inference
        self.logger.debug("Loading ONNX model for inference...")
        if self.segment_size == self.dim_t:
            ort_ = ort.InferenceSession(model_path, providers=self.onnx_execution_provider)
            self.model_run = lambda spek: ort_.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Model loaded successfully using ONNXruntime inferencing session.")
        else:
            self.model_run = convert(model_path)
            self.model_run.to(self.device).eval()
            self.logger.warning("Model converted from onnx to pytorch due to segment size not matching dim_t, processing may be slower.")

        # Log the completion of the separation process
        self.logger.debug("Loading model completed.")
        self.logger.info(
            f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - self.load_model_start_time)))}'
        )

    def separate(self, audio_file_path):
        # Starting the separation process
        self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
        self.separate_start_time = time.perf_counter()

        self.primary_source = None
        self.secondary_source = None

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        # Prepare the mix for processing
        self.logger.debug("Preparing mix...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug("Normalizing mix before demixing...")
        mix = spec_utils.normalize(self.logger, wave=mix, max_peak=self.normalization_threshold)

        # Start the demixing process
        source = self.demix(mix)

        # In UVR, the source is cached here if it's a vocal split model, but we're not supporting that yet

        # Initialize the list for output files
        output_files = []
        self.logger.debug("Processing output files...")

        # Normalize and transpose the primary source if it's not already an array
        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug("Normalizing primary source...")
            self.primary_source = spec_utils.normalize(self.logger, wave=source, max_peak=self.normalization_threshold).T

        # Process the secondary source if not already an array
        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug("Producing secondary source: demixing in match_mix mode")
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                self.logger.debug("Inverting secondary stem using spectogram as invert_using_spec is set to True")
                self.secondary_source = spec_utils.invert_stem(raw_mix, source)
            else:
                self.logger.debug("Inverting secondary stem by subtracting of transposed demixed stem from transposed original mix")
                self.secondary_source = mix.T - source.T

        # Save and process the secondary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.model_secondary_stem.lower():
            self.logger.info(f"Saving {self.model_secondary_stem} stem...")
            if not self.secondary_stem_path:
                self.secondary_stem_path = os.path.join(
                    f"{self.audio_file_base}_({self.model_secondary_stem})_{self.model_name}.{self.output_format.lower()}"
                )
            self.secondary_source_map = self.final_process(
                self.secondary_stem_path, self.secondary_source, self.model_secondary_stem, self.sample_rate
            )
            output_files.append(self.secondary_stem_path)

        # Save and process the primary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.model_primary_stem.lower():
            self.logger.info(f"Saving {self.model_primary_stem} stem...")
            if not self.primary_stem_path:
                self.primary_stem_path = os.path.join(
                    f"{self.audio_file_base}_({self.model_primary_stem})_{self.model_name}.{self.output_format.lower()}"
                )
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T
            self.primary_source_map = self.final_process(
                self.primary_stem_path, self.primary_source, self.model_primary_stem, self.sample_rate
            )
            output_files.append(self.primary_stem_path)

        # Clear GPU cache to free up memory
        self.clear_gpu_cache()

        # TODO: In UVR, this is where the vocal split chain gets processed - see process_vocal_split_chain()

        # Log the completion of the separation process
        self.logger.debug("Separation process completed.")
        self.logger.info(
            f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - self.separate_start_time)))}'
        )

        # Unset the audio file to prevent accidental re-separation of the same file
        self.logger.debug("Clearing audio file...")
        self.audio_file_path = None
        self.audio_file_base = None

        # Unset more separation params to prevent accidentally re-using the wrong source files or output paths
        self.logger.debug("Clearing sources and stems...")
        self.primary_source = None
        self.secondary_source = None
        self.primary_stem_path = None
        self.secondary_stem_path = None

        return output_files

    def write_audio(self, stem_path: str, stem_source, sample_rate, stem_name=None):
        self.logger.debug(f"Entering write_audio with stem_name: {stem_name} and stem_path: {stem_path}")

        stem_source = spec_utils.normalize(self.logger, wave=stem_source, max_peak=self.normalization_threshold)

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
                stem_source_interleaved.tobytes(), frame_rate=self.sample_rate, sample_width=stem_source.dtype.itemsize, channels=2
            )
            self.logger.debug("Created AudioSegment successfully.")
        except Exception as e:
            self.logger.error(f"Error creating AudioSegment: {e}")
            return

        # Determine file format based on the file extension
        file_format = stem_path.lower().split(".")[-1]

        # For m4a files, specify mp4 as the container format as the extension doesn't match the format name
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

        # chunk_size is the hop_length size times the segment size minus one
        self.chunk_size = self.hop_length * (self.segment_size - 1)

        # gen_size is the chunk size minus twice the trim size
        self.gen_size = self.chunk_size - 2 * self.trim

        self.stft = STFT(self.logger, self.n_fft, self.hop_length, self.dim_f, self.device)

        self.logger.debug(f"Model input params: n_fft={self.n_fft} hop_length={self.hop_length} dim_f={self.dim_f}")
        self.logger.debug(f"Model settings: n_bins={self.n_bins}, trim={self.trim}, chunk_size={self.chunk_size}, gen_size={self.gen_size}")

    # After prepare_mix segments the audio, initialize_mix further processes each segment.
    # It ensures each audio segment is in the correct format for the model, applies necessary padding,
    # and converts the segments into tensors for processing with the model.
    # This step is essential for preparing the audio data in a format that the neural network can process.
    def initialize_mix(self, mix, is_ckpt=False):
        # Log the initialization of the mix and whether checkpoint mode is used
        self.logger.debug(f"Initializing mix with is_ckpt={is_ckpt}. Initial mix shape: {mix.shape}")

        # Ensure the mix is a 2-channel (stereo) audio signal
        if mix.shape[0] != 2:
            error_message = f"Expected a 2-channel audio signal, but got {mix.shape[0]} channels"
            self.logger.error(error_message)
            raise ValueError(error_message)

        # If in checkpoint mode, process the mix differently
        if is_ckpt:
            self.logger.debug("Processing in checkpoint mode...")
            # Calculate padding based on the generation size and trim
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            self.logger.debug(f"Padding calculated: {pad}")
            # Add padding at the beginning and the end of the mix
            mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)
            # Determine the number of chunks based on the mixture's length
            num_chunks = mixture.shape[-1] // self.gen_size
            self.logger.debug(f"Mixture shape after padding: {mixture.shape}, Number of chunks: {num_chunks}")
            # Split the mixture into chunks
            mix_waves = [mixture[:, i * self.gen_size : i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            # If not in checkpoint mode, process normally
            self.logger.debug("Processing in non-checkpoint mode...")
            mix_waves = []
            n_sample = mix.shape[1]
            # Calculate necessary padding to make the total length divisible by the generation size
            pad = self.gen_size - n_sample % self.gen_size
            self.logger.debug(f"Number of samples: {n_sample}, Padding calculated: {pad}")
            # Apply padding to the mix
            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
            self.logger.debug(f"Shape of mix after padding: {mix_p.shape}")

            # Process the mix in chunks
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + self.chunk_size])
                mix_waves.append(waves)
                self.logger.debug(f"Processed chunk {len(mix_waves)}: Start {i}, End {i + self.chunk_size}")
                i += self.gen_size

        # Convert the list of wave chunks into a tensor for processing on the specified device
        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        self.logger.debug(f"Converted mix_waves to tensor. Tensor shape: {mix_waves_tensor.shape}")

        return mix_waves_tensor, pad

    def demix(self, mix, is_match_mix=False):
        self.logger.debug(f"Starting demixing process with is_match_mix: {is_match_mix}...")
        self.initialize_model_settings()

        # Preserves the original mix for later use.
        # In UVR, this is used for the pitch fix and VR denoise processes, which aren't yet implemented here.
        org_mix = mix
        self.logger.debug(f"Original mix stored. Shape: {org_mix.shape}")

        # Initializes a list to store the separated waveforms.
        tar_waves_ = []

        # Handling different chunk sizes and overlaps based on the matching requirement.
        if is_match_mix:
            # Sets a smaller chunk size specifically for matching the mix.
            chunk_size = self.hop_length * (self.segment_size - 1)
            # Sets a small overlap for the chunks.
            overlap = 0.02
            self.logger.debug(f"Chunk size for matching mix: {chunk_size}, Overlap: {overlap}")
        else:
            # Uses the regular chunk size defined in model settings.
            chunk_size = self.chunk_size
            # Uses the overlap specified in the model settings.
            overlap = self.overlap
            self.logger.debug(f"Standard chunk size: {chunk_size}, Overlap: {overlap}")

        # Calculates the generated size after subtracting the trim from both ends of the chunk.
        gen_size = chunk_size - 2 * self.trim
        self.logger.debug(f"Generated size calculated: {gen_size}")

        # Calculates padding to make the mix length a multiple of the generated size.
        pad = gen_size + self.trim - ((mix.shape[-1]) % gen_size)
        # Prepares the mixture with padding at the beginning and the end.
        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)
        self.logger.debug(f"Mixture prepared with padding. Mixture shape: {mixture.shape}")

        # Calculates the step size for processing chunks based on the overlap.
        step = int((1 - overlap) * chunk_size)
        self.logger.debug(f"Step size for processing chunks: {step} as overlap is set to {overlap}.")

        # Initializes arrays to store the results and to account for overlap.
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        # Initializes counters for processing chunks.
        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step
        self.logger.debug(f"Total chunks to process: {total_chunks}")

        # Processes each chunk of the mixture.
        for i in range(0, mixture.shape[-1], step):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            self.logger.debug(f"Processing chunk {total}/{total_chunks}: Start {start}, End {end}")

            # Handles windowing for overlapping chunks.
            chunk_size_actual = end - start
            window = None
            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))
                self.logger.debug("Window applied to the chunk.")

            # Zero-pad the chunk to prepare it for processing.
            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype="float32")), axis=-1)

            # Converts the chunk to a tensor for processing.
            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.device)
            # Splits the chunk into smaller batches if necessary.
            mix_waves = mix_part.split(self.batch_size)
            total_batches = len(mix_waves)
            self.logger.debug(f"Mix part split into batches. Number of batches: {total_batches}")

            with torch.no_grad():
                # Processes each batch in the chunk.
                batches_processed = 0
                for mix_wave in mix_waves:
                    batches_processed += 1
                    self.logger.debug(f"Processing mix_wave batch {batches_processed}/{total_batches}")

                    # Runs the model to separate the sources.
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    # Applies windowing if needed and accumulates the results.
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]

        # Normalizes the results by the divider to account for overlap.
        self.logger.debug("Normalizing result by dividing result by divider.")
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        # Reshapes the results to match the original dimensions.
        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim : -self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, : mix.shape[-1]]

        # Extracts the source from the results.
        source = tar_waves[:, 0:None]
        self.logger.debug(f"Concatenated tar_waves. Shape: {tar_waves.shape}")

        # TODO: In UVR, pitch changing happens here. Consider implementing this as a feature.

        # Compensates the source if not matching the mix.
        if not is_match_mix:
            source * self.compensate
            self.logger.debug("Match mix mode; compensate multiplier applied.")

        # TODO: In UVR, VR denoise model gets applied here. Consider implementing this as a feature.

        self.logger.debug("Demixing process completed.")
        return source

    def run_model(self, mix, is_match_mix=False):
        # Applying the STFT to the mix. The mix is moved to the specified device (e.g., GPU) before processing.
        # self.logger.debug(f"Running STFT on the mix. Mix shape before STFT: {mix.shape}")
        spek = self.stft(mix.to(self.device))
        self.logger.debug(f"STFT applied on mix. Spectrum shape: {spek.shape}")

        # Zeroing out the first 3 bins of the spectrum. This is often done to reduce low-frequency noise.
        spek[:, :, :3, :] *= 0
        # self.logger.debug("First 3 bins of the spectrum zeroed out.")

        # Handling the case where the mix needs to be matched (is_match_mix = True)
        if is_match_mix:
            # self.logger.debug("Match mix mode is enabled. Converting spectrum to NumPy array.")
            spec_pred = spek.cpu().numpy()
            self.logger.debug("is_match_mix: spectrum prediction obtained directly from STFT output.")
        else:
            # If denoising is enabled, the model is run on both the negative and positive spectrums.
            if self.denoise_enabled:
                spec_pred = -self.model_run(-spek) * 0.5 + self.model_run(spek) * 0.5
                self.logger.debug("Model run on both negative and positive spectrums for denoising.")
            else:
                spec_pred = self.model_run(spek)
                self.logger.debug("Model run on the spectrum without denoising.")

        # Applying the inverse STFT to convert the spectrum back to the time domain.
        result = self.stft.inverse(torch.tensor(spec_pred).to(self.device)).cpu().detach().numpy()
        self.logger.debug(f"Inverse STFT applied. Returning result with shape: {result.shape}")

        return result

    def prepare_mix(self, mix):
        # Store the original path or the mix itself for later checks
        audio_path = mix

        # Check if the input is a file path (string) and needs to be loaded
        if not isinstance(mix, np.ndarray):
            self.logger.debug(f"Loading audio from file: {mix}")
            mix, sr = librosa.load(mix, mono=False, sr=self.sample_rate)
            self.logger.debug(f"Audio loaded. Sample rate: {sr}, Audio shape: {mix.shape}")
        else:
            # Transpose the mix if it's already an ndarray (expected shape: [channels, samples])
            self.logger.debug("Transposing the provided mix array.")
            mix = mix.T
            self.logger.debug(f"Transposed mix shape: {mix.shape}")

        # If the original input was a filepath, check if the loaded mix is empty
        if isinstance(audio_path, str):
            if not np.any(mix):
                error_msg = f"Audio file {audio_path} is empty or not valid"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("Audio file is valid and contains data.")

        # Ensure the mix is in stereo format
        if mix.ndim == 1:
            self.logger.debug("Mix is mono. Converting to stereo.")
            mix = np.asfortranarray([mix, mix])
            self.logger.debug("Converted to stereo mix.")

        # Final log indicating successful preparation of the mix
        self.logger.debug("Mix preparation completed.")
        return mix
