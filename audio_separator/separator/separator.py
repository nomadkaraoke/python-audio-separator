""" This file contains the Separator class, to facilitate the separation of stems from audio. """

from importlib import metadata
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
import onnxruntime as ort
from tqdm import tqdm
from audio_separator.separator.architectures import MDXSeparator, VRSeparator


class Separator:
    """
    The Separator class is designed to facilitate the separation of audio sources from a given audio file.
    It supports various separation architectures and models, including MDX and VR. The class provides
    functionalities to configure separation parameters, load models, and perform audio source separation.
    It also handles logging, normalization, and output formatting of the separated audio stems.

    Common Attributes:
        log_level (int): The logging level.
        log_formatter (logging.Formatter): The logging formatter.
        model_file_dir (str): The directory where model files are stored.
        output_dir (str): The directory where output files will be saved.
        primary_stem_path (str): The path for saving the primary stem.
        secondary_stem_path (str): The path for saving the secondary stem.
        output_format (str): The format of the output audio file.
        output_subtype (str): The subtype of the output audio format.
        normalization_threshold (float): The threshold for audio normalization.
        denoise_enabled (bool): Flag to enable or disable denoising.
        output_single_stem (str): Option to output a single stem.
        invert_using_spec (bool): Flag to invert using spectrogram.
        sample_rate (int): The sample rate of the audio.

    MDX Model Specific Attributes:
        hop_length (int): The hop length for STFT.
        segment_size (int): The segment size for processing.
        overlap (float): The overlap between segments.
        batch_size (int): The batch size for processing.
    """

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
                raise ValueError("output_single_stem must be either 'instrumental' or 'vocals'")
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

        self.torch_device = None
        self.onnx_execution_provider = None
        self.model_instance = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None

        self.setup_accelerated_inferencing_device()

    def setup_accelerated_inferencing_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        self.log_system_info()
        self.log_onnxruntime_packages()
        self.setup_torch_device()

    def log_system_info(self):
        """
        This method logs the system information, including the operating system, CPU archutecture and Python version
        """
        os_name = platform.system()
        os_version = platform.version()
        self.logger.info(f"Operating System: {os_name} {os_version}")

        system_info = platform.uname()
        self.logger.info(f"System: {system_info.system} Node: {system_info.node} Release: {system_info.release} Machine: {system_info.machine} Proc: {system_info.processor}")

        python_version = platform.python_version()
        self.logger.info(f"Python Version: {python_version}")

    def log_onnxruntime_packages(self):
        """
        This method logs the ONNX Runtime package versions, including the GPU and Silicon packages if available.
        """
        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        onnxruntime_silicon_package = self.get_package_distribution("onnxruntime-silicon")
        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")

        if onnxruntime_gpu_package is not None:
            self.logger.info(f"ONNX Runtime GPU package installed with version: {onnxruntime_gpu_package.version}")
        if onnxruntime_silicon_package is not None:
            self.logger.info(f"ONNX Runtime Silicon package installed with version: {onnxruntime_silicon_package.version}")
        if onnxruntime_cpu_package is not None:
            self.logger.info(f"ONNX Runtime CPU package installed with version: {onnxruntime_cpu_package.version}")

    def setup_torch_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        hardware_acceleration_enabled = False
        ort_providers = ort.get_available_providers()

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
            self.torch_device = torch.device("cpu")
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        """
        This method configures the CUDA device for PyTorch and ONNX Runtime, if available.
        """
        self.logger.info("CUDA is available in Torch, setting Torch device to CUDA")
        self.torch_device = torch.device("cuda")
        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime has CUDAExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
        else:
            self.logger.warning("CUDAExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def configure_mps(self, ort_providers):
        """
        This method configures the Apple Silicon MPS/CoreML device for PyTorch and ONNX Runtime, if available.
        """
        self.logger.info("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
        self.logger.warning("Torch MPS backend does not yet support FFT operations, Torch will still use CPU!")
        self.torch_device = torch.device("cpu")
        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime has CoreMLExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else:
            self.logger.warning("CoreMLExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def get_package_distribution(self, package_name):
        """
        This method returns the package distribution for a given package name if installed, or None otherwise.
        """
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(f"Python package: {package_name} not installed")
            return None

    def get_model_hash(self, model_path):
        """
        This method returns the MD5 hash of a given model file.
        """

        self.logger.error(f"Attempting to calculate hash of model file {model_path}")
        try:
            # Open the model file in binary read mode
            with open(model_path, "rb") as f:
                # Move the file pointer 10MB before the end of the file
                f.seek(-10000 * 1024, 2)
                # Read the file from the current pointer to the end and calculate its MD5 hash
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            # If an IOError occurs (e.g., file not found), log the error
            self.logger.error(f"IOError reading model file for hash calculation: {e}")
            # Attempt to open the file again, read its entire content, and calculate the MD5 hash
            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file(self, url, output_path):
        """
        This method downloads a file from a given URL to a given output path.
        """
        self.logger.debug(f"Downloading file from {url} to {output_path} with timeout 300s and verify=False")
        response = requests.get(url, stream=True, timeout=300, verify=False)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            self.logger.error(f"Failed to download file from {url}")

    def clear_gpu_cache(self):
        """
        This method clears the GPU cache to free up memory.
        """
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if self.torch_device == torch.device("mps"):
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if self.torch_device == torch.device("cuda"):
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()

    def load_model(self, model_filename="UVR-MDX-NET-Inst_HQ_3.onnx", model_type=None):
        """
        This method loads the separation model into memory, downloading it first if necessary.
        """
        self.logger.info(f"Loading model {model_filename}...")

        load_model_start_time = time.perf_counter()

        model_name = model_filename.split(".")[0]
        model_url = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{model_filename}"
        model_data_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"

        # Setting up the model path
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")
        self.logger.debug(f"Model path set to {model_path}")

        # Check if model file exists, if not, download it
        if not os.path.isfile(model_path):
            self.logger.debug(f"Model not found at path {model_path}, downloading...")
            self.download_file(model_url, model_path)

        # Reading model settings from the downloaded model
        self.logger.debug("Reading model settings...")
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(f"Model {model_path} has hash {model_hash}")

        # Setting up the path for model data and checking its existence
        model_data_path = os.path.join(self.model_file_dir, "model_data.json")
        self.logger.debug(f"Model data path set to {model_data_path}")
        if not os.path.isfile(model_data_path):
            self.logger.debug(f"Model data not found at path {model_data_path}, downloading...")
            self.download_file(model_data_url, model_data_path)

        # Loading model data
        self.logger.debug("Loading model data...")
        model_data_object = json.load(open(model_data_path, encoding="utf-8"))
        model_data = model_data_object[model_hash]
        self.logger.debug(f"Model data loaded: {model_data}")

        # Identify model_type based on the file extension
        file_extension = model_filename.split(".")[-1]
        if file_extension == "onnx":
            model_type = "MDX"
        elif file_extension == "pth":
            model_type = "VR"
        else:
            raise ValueError(f"Unsupported model file extension: {file_extension}")

        common_params = {
            "logger": self.logger,
            "torch_device": self.torch_device,
            "onnx_execution_provider": self.onnx_execution_provider,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "primary_stem_path": self.primary_stem_path,
            "secondary_stem_path": self.secondary_stem_path,
            "output_format": self.output_format,
            "output_subtype": self.output_subtype,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "denoise_enabled": self.denoise_enabled,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
        }

        arch_specific_params = {"hop_length": self.hop_length, "segment_size": self.segment_size, "overlap": self.overlap, "batch_size": self.batch_size}

        if model_type == "MDX":
            self.model_instance = MDXSeparator(common_config=common_params, arch_config=arch_specific_params)
        elif model_type == "VR":
            self.model_instance = VRSeparator(common_config=common_params, arch_config=arch_specific_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Log the completion of the model load process
        self.logger.debug("Loading model completed.")
        self.logger.info(f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate(self, audio_file_path):
        """
        Separates the audio file into different stems (e.g., vocals, instruments) using the loaded model.

        This method takes the path to an audio file, processes it through the loaded separation model, and returns
        the paths to the output files containing the separated audio stems. It handles the entire flow from loading
        the audio, running the separation, clearing up resources, and logging the process.

        Parameters:
        - audio_file_path (str): The path to the audio file to be separated.

        Returns:
        - output_files (list of str): A list containing the paths to the separated audio stem files.
        """
        # Starting the separation process
        self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
        separate_start_time = time.perf_counter()

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
        self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}')

        return output_files
