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
import onnxruntime as ort
import numpy as np
from importlib import metadata
from pydub import AudioSegment
from audio_separator.separator import spec_utils
from audio_separator.separator.architectures import MDXSeparator


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
        self.logger.debug(f"Normalization threshold set to {normalization_threshold}, waveform will lowered to this max amplitude to avoid clipping.")

        self.denoise_enabled = denoise_enabled
        if self.denoise_enabled:
            self.logger.debug(f"Denoising enabled, model will be run twice to reduce noise in output audio.")
        else:
            self.logger.debug(f"Denoising disabled, model will only be run once. This is twice as fast, but may result in noisier output audio.")

        self.output_single_stem = output_single_stem
        if output_single_stem is not None:
            if output_single_stem.lower() not in {"instrumental", "vocals"}:
                raise Exception("output_single_stem must be either 'instrumental' or 'vocals'")
            self.logger.debug(f"Single stem output requested, only one output file ({output_single_stem}) will be written")

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec:
            self.logger.debug(f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality, but is slightly slower.")

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
        self.logger.info(f"System: {system_info.system} Node: {system_info.node} Release: {system_info.release} Machine: {system_info.machine} Proc: {system_info.processor}")

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

    def clear_gpu_cache(self):
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if self.device == torch.device("mps"):
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if self.device == torch.device("cuda"):
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()

    def load_model(self, model_filename="UVR-MDX-NET-Inst_HQ_3.onnx", model_type="MDX"):
        self.logger.info(f"Loading model {model_filename}...")

        self.load_model_start_time = time.perf_counter()

        self.model_filename = model_filename
        self.model_name = self.model_filename.split(".")[0]

        self.model_url = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{self.model_filename}"
        self.model_data_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"

        # Setting up the model path
        model_path = os.path.join(self.model_file_dir, f"{self.model_filename}")
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

        separator_params = {
            "model_name": self.model_name,
            "model_path": model_path,
            "model_data": model_data,
            "primary_stem_path": self.primary_stem_path,
            "secondary_stem_path": self.secondary_stem_path,
            "output_format": self.output_format,
            "output_subtype": self.output_subtype,
            "normalization_threshold": self.normalization_threshold,
            "denoise_enabled": self.denoise_enabled,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "segment_size": self.segment_size,
            "overlap": self.overlap,
            "batch_size": self.batch_size,
            "device": self.device,
            "onnx_execution_provider": self.onnx_execution_provider,
        }

        self.model_type = model_type

        if self.model_type == "MDX":
            self.model_instance = MDXSeparator(logger=self.logger, write_audio=self.write_audio, separator_params=separator_params)
        elif self.model_type == "VR":
            self.model_instance = VRSeparator(model_data=model_data)
        elif self.model_type == "Demucs":
            self.model_instance = DemucsSeparator(model_data=model_data)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Log the completion of the model load process
        self.logger.debug("Loading model completed.")
        self.logger.info(f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - self.load_model_start_time)))}')

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
            audio_segment = AudioSegment(stem_source_interleaved.tobytes(), frame_rate=self.sample_rate, sample_width=stem_source.dtype.itemsize, channels=2)
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

    def separate(self, audio_file_path):
        # Starting the separation process
        self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
        self.separate_start_time = time.perf_counter()

        # Run separation method for the loaded model
        output_files = self.model_instance.separate(audio_file_path)

        # Clear GPU cache to free up memory
        self.clear_gpu_cache()

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

        # Log the completion of the separation process
        self.logger.debug("Separation process completed.")
        self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - self.separate_start_time)))}')

        return output_files
