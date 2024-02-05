# Audio Separator 🎶

[![PyPI version](https://badge.fury.io/py/audio-separator.svg)](https://badge.fury.io/py/audio-separator)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/audio-separator.svg)](https://anaconda.org/conda-forge/audio-separator)
[![Docker pulls](https://img.shields.io/docker/pulls/beveradb/audio-separator.svg)](https://hub.docker.com/r/beveradb/audio-separator/tags)

Summary: Easy to use audio stem separation from the command line or as a dependency in your own Python project, using the amazing MDX-Net and VR Arch models available in UVR by @Anjok07 & @aufr33.

Audio Separator is a Python package that allows you to separate an audio file into various stems, using models trained by @Anjok07 for use with UVR (https://github.com/Anjok07/ultimatevocalremovergui).

The simplest (and probably most utilized) use case for this package is to separate an audio file into two stems, Instrumental and Vocals which can be very useful for producing Karaoke videos! However, the models available in UVR can separate audio into many more stems, such as Drums, Bass, Piano, Guitar, and perform other audio processing tasks such as denoising and removing echo / reverb.

## Features

- Separate audio into multiple stems, e.g. instrumental and vocals.
- Supports all common audio formats (WAV, MP3, FLAC, M4A, etc.)
- Ability to inference using a pre-trained model in PTH or ONNX format.
- CLI support for easy use in scripts and batch processing.
- Python API for integration into other projects.

## Installation 🛠️

### 🎮 Nvidia GPU with CUDA or 🧪 Google Colab

💬 If successfully configured, you should see this log message when running `audio-separator --env_info`:
 `ONNXruntime has CUDAExecutionProvider available, enabling acceleration`

Conda: `conda install pytorch=*=*cuda* onnxruntime=*=*cuda* audio-separator -c pytorch -c conda-forge`

Pip: `pip install "audio-separator[gpu]"`

Docker: `beveradb/audio-separator:gpu`

###  Apple Silicon, macOS Sonoma+ with M1 or newer CPU (CoreML acceleration)

💬 If successfully configured, you should see this log message when running `audio-separator --env_info`:
 `ONNXruntime has CoreMLExecutionProvider available, enabling acceleration`

Pip: `pip install "audio-separator[cpu]"`

### 🐢 No hardware acceleration, CPU only:

Conda: `conda install audio-separator-c pytorch -c conda-forge`

Pip: `pip install "audio-separator[cpu]"`

Docker: `beveradb/audio-separator`


### 🎥 FFmpeg dependency

💬 If successfully configured, you should see a `FFmpeg installed` log message when running `audio-separator --env_info`

If you installed `audio-separator` using `conda` or `docker`, FFmpeg should already be avaialble in your environment.

If not, you'll separately need to ensure you have `ffmpeg` installed.
This should be easy to install on most platforms, e.g.:

🐧 Debian/Ubuntu: `apt-get update; apt-get install -y ffmpeg`

 macOS:`brew update; brew install ffmpeg`


## GPU / CUDA specific installation steps with Pip

In theory, all you should need to do to get `audio-separator` working with a GPU is install it with the `[gpu]` extra as above.

However, sometimes getting both PyTorch and ONNX Runtime working with CUDA support can be a bit tricky so it may not work that easily.

You may need to reinstall both packages directly, allowing pip to calculate the right versions for your platform:

- `pip uninstall torch onnxruntime`
- `pip cache purge`
- `pip install --force-reinstall torch torchvision torchaudio`
- `pip install --force-reinstall onnxruntime-gpu`

Depending on your hardware, you may get better performance with the optimum version of onnxruntime:
- `pip install --force-reinstall "optimum[onnxruntime-gpu]"`

Depending on your CUDA version and hardware, you may need to install torch from the `cu118` index instead:
- `pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

> Note: if anyone knows how to make this cleaner so we can support both different platform-specific dependencies for hardware acceleration without a separate installation process for each, please let me know or raise a PR!

## Usage in Docker 🐳

There are [images published on Docker Hub](https://hub.docker.com/r/beveradb/audio-separator/tags) for GPU (CUDA) and CPU inferencing, for both `amd64` and `arm64` platforms.

You probably want to volume-mount a folder containing whatever file you want to separate, which can then also be used as the output folder.

For example, if the current directory contains your input file `input.wav`, you could run `audio-separator` like so:

```
docker run -it -v `pwd`:/workdir beveradb/audio-separator input.wav
```

If you're using a machine with a GPU, you'll want to use the GPU specific image and pass in the GPU device to the container, like this:

```
docker run -it --gpus all -v `pwd`:/workdir beveradb/audio-separator:gpu input.wav
```

If the GPU isn't being detected, make sure your docker runtime environment is passing through the GPU correctly - there are [various guides](https://www.celantur.com/blog/run-cuda-in-docker-on-linux/) online to help with that.


## Usage 🚀

### Command Line Interface (CLI)

You can use Audio Separator via the command line:

```sh
usage: audio-separator [-h] [-v] [-d] [-e] [-m] [--log_level LOG_LEVEL] [--model_filename MODEL_FILENAME] [--output_format OUTPUT_FORMAT] [--output_dir OUTPUT_DIR] [--model_file_dir MODEL_FILE_DIR] [--denoise]
                       [--invert_spect] [--normalization NORMALIZATION] [--single_stem SINGLE_STEM] [--sample_rate SAMPLE_RATE] [--mdx_segment_size MDX_SEGMENT_SIZE] [--mdx_overlap MDX_OVERLAP]
                       [--mdx_batch_size MDX_BATCH_SIZE] [--mdx_hop_length MDX_HOP_LENGTH] [--vr_batch_size VR_BATCH_SIZE] [--vr_window_size VR_WINDOW_SIZE] [--vr_aggression VR_AGGRESSION] [--vr_enable_tta]
                       [--vr_high_end_process] [--vr_enable_post_process] [--vr_post_process_threshold VR_POST_PROCESS_THRESHOLD]
                       [audio_file]

Separate audio file into different stems.

positional arguments:
  audio_file                                             The audio file path to separate, in any common format.

options:
  -h, --help                                             show this help message and exit

Info and Debugging:
  -v, --version                                          show program's version number and exit
  -d, --debug                                            enable debug logging, equivalent to --log_level=debug
  -e, --env_info                                         print environment information and exit.
  -m, --list_models                                      list all supported models and exit.
  --log_level LOG_LEVEL                                  log level, e.g. info, debug, warning (default: info)

Separation I/O Params:
  --model_filename MODEL_FILENAME                        model to use for separation (default: 2_HP-UVR.pth). Example: --model_filename=UVR_MDXNET_KARA_2.onnx
  --output_format OUTPUT_FORMAT                          output format for separated files, any common format (default: FLAC). Example: --output_format=MP3
  --output_dir OUTPUT_DIR                                directory to write output files (default: <current dir>). Example: --output_dir=/app/separated
  --model_file_dir MODEL_FILE_DIR                        model files directory (default: /tmp/audio-separator-models/). Example: --model_file_dir=/app/models

Common Separation Parameters:
  --denoise                                              enable denoising during separation (default: False). Example: --denoise
  --invert_spect                                         invert secondary stem using spectogram (default: False). Example: --invert_spect
  --normalization NORMALIZATION                          max peak amplitude to normalize input and output audio to (default: 0.9). Example: --normalization=0.7
  --single_stem SINGLE_STEM                              output only single stem, either instrumental or vocals. Example: --single_stem=instrumental
  --sample_rate SAMPLE_RATE                              modify the sample rate of the output audio (default: 44100). Example: --sample_rate=44100

MDX Architecture Parameters:
  --mdx_segment_size MDX_SEGMENT_SIZE                    larger consumes more resources, but may give better results (default: 256). Example: --mdx_segment_size=256
  --mdx_overlap MDX_OVERLAP                              amount of overlap between prediction windows, 0.001-0.999. higher is better but slower (default: 0.25). Example: --mdx_overlap=0.25
  --mdx_batch_size MDX_BATCH_SIZE                        larger consumes more RAM but may process slightly faster (default: 1). Example: --mdx_batch_size=4
  --mdx_hop_length MDX_HOP_LENGTH                        usually called stride in neural networks, only change if you know what you're doing (default: 1024). Example: --mdx_hop_length=1024

VR Architecture Parameters:
  --vr_batch_size VR_BATCH_SIZE                          number of batches to process at a time. higher = more RAM, slightly faster processing (default: 4). Example: --vr_batch_size=16
  --vr_window_size VR_WINDOW_SIZE                        balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --vr_window_size=320
  --vr_aggression VR_AGGRESSION                          intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: 5). Example: --vr_aggression=2
  --vr_enable_tta                                        enable Test-Time-Augmentation; slow but improves quality (default: False). Example: --vr_enable_tta
  --vr_high_end_process                                  mirror the missing frequency range of the output (default: False). Example: --vr_high_end_process
  --vr_enable_post_process                               identify leftover artifacts within vocal output; may improve separation for some songs (default: False). Example: --vr_enable_post_process
  --vr_post_process_threshold VR_POST_PROCESS_THRESHOLD  threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --vr_post_process_threshold=0.1
```

Example:

```
audio-separator /path/to/your/audio.wav --model_name UVR_MDXNET_KARA_2
```

This command will process the file and generate two new files in the current directory, one for each stem.

### As a Dependency in a Python Project

You can use Audio Separator in your own Python project. Here's how you can use it:

```
from audio_separator.separator import Separator

# Initialize the Separator class (with optional configuration properties below)
separator = Separator()

# Load a machine learning model (if unspecified, defaults to 'UVR-MDX-NET-Inst_HQ_3.onnx')
separator.load_model()

# Perform the separation on specific audio files without reloading the model
primary_stem_output_path, secondary_stem_output_path = separator.separate('audio1.wav')

print(f'Primary stem saved at {primary_stem_output_path}')
print(f'Secondary stem saved at {secondary_stem_output_path}')
```

#### Batch processing, or processing with multiple models

You can process multiple separations without reloading the model, to save time and memory.

You only need to load a model when choosing or changing models. See example below:

```
from audio_separator.separator import Separator

# Initialize the Separator with other configuration properties below
separator = Separator()

# Load a model
separator.load_model('UVR-MDX-NET-Inst_HQ_3.onnx')

# Separate multiple audio files without reloading the model
output_file_paths_1 = separator.separate('audio1.wav')
output_file_paths_2 = separator.separate('audio2.wav')
output_file_paths_3 = separator.separate('audio3.wav')

# Load a different model
separator.load_model('UVR_MDXNET_KARA_2')

# Separate the same files with the new model
output_file_paths_4 = separator.separate('audio1.wav')
output_file_paths_5 = separator.separate('audio2.wav')
output_file_paths_6 = separator.separate('audio3.wav')
```

## Parameters for the Separator class

- log_level: (Optional) Logging level, e.g., INFO, DEBUG, WARNING. Default: DEBUG
- log_formatter: (Optional) The log format. Default: None, which falls back to '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
- model_file_dir: (Optional) Directory to cache model files in. Default: /tmp/audio-separator-models/
- output_dir: (Optional) Directory where the separated files will be saved. If not specified, uses the current directory.
- primary_stem_output_path: (Optional) The path for saving the primary stem. Default: None
- secondary_stem_output_path: (Optional) The path for saving the secondary stem. Default: None
- output_format: (Optional) Format to encode output files, any common format (WAV, MP3, FLAC, M4A, etc.). Default: WAV
- normalization_threshold: (Optional) The threshold for audio normalization. Default: 0.9
- enable_denoise: (Optional) Flag to enable or disable denoising as part of the separation process. Default: False
- output_single_stem: (Optional) Output only a single stem, either 'instrumental' or 'vocals'. Default: None
- invert_using_spec: (Optional) Flag to invert using spectrogram. Default: False
- sample_rate: (Optional) Modify the sample rate of the output audio. Default: 44100
- mdx_params: (Optional) MDX Architecture Specific Attributes & Defaults. Default: {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1}
- vr_params: (Optional) VR Architecture Specific Attributes & Defaults. Default: {"batch_size": 16, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}

## Requirements 📋

Python >= 3.9

Libraries: torch, onnx, onnxruntime, numpy, librosa, requests, six, tqdm, pydub

## Developing Locally

This project uses Poetry for dependency management and packaging. Follow these steps to setup a local development environment:

### Prerequisites

- Make sure you have Python 3.9 or newer installed on your machine.
- Install Conda (I recommend Miniforge: https://github.com/conda-forge/miniforge) to manage your Python virtual environments

### Clone the Repository

Clone the repository to your local machine:

```
git clone https://github.com/YOUR_USERNAME/audio-separator.git
cd audio-separator
```

Replace YOUR_USERNAME with your GitHub username if you've forked the repository, or use the main repository URL if you have the permissions.

### Create and activate the Conda Environment

To create and activate the conda environment, use the following commands:

```
conda env create
conda activate audio-separator-dev
```

### Install Dependencies

Once you're inside the conda env, run the following command to install the project dependencies:

```
poetry install
```


### Running the Command-Line Interface Locally

You can run the CLI command directly within the virtual environment. For example:

```
audio-separator path/to/your/audio-file.wav
```

### Deactivate the Virtual Environment

Once you are done with your development work, you can exit the virtual environment by simply typing:

```
conda deactivate
```

### Building the Package

To build the package for distribution, use the following command:

```
poetry build
```

This will generate the distribution packages in the dist directory - but for now only @beveradb will be able to publish to PyPI.

## Contributing 🤝

Contributions are very much welcome! Please fork the repository and submit a pull request with your changes, and I'll try to review, merge and publish promptly!

- This project is 100% open-source and free for anyone to use and modify as they wish. 
- If the maintenance workload for this repo somehow becomes too much for me I'll ask for volunteers to share maintainership of the repo, though I don't think that is very likely
- Development and support for the MDX-Net separation models is part of the main [UVR project](https://github.com/Anjok07/ultimatevocalremovergui), this repo is just a CLI/Python package wrapper to simplify running those models programmatically. So, if you want to try and improve the actual models, please get involved in the UVR project and look for guidance there!

## License 📄

This project is licensed under the MIT [License](LICENSE).

- **Please Note:** If you choose to integrate this project into some other project using the default model or any other model trained as part of the [UVR](https://github.com/Anjok07/ultimatevocalremovergui) project, please honor the MIT license by providing credit to UVR and its developers!

## Credits 🙏

- [Anjok07](https://github.com/Anjok07) - Author of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui), which almost all of the code in this repo was copied from! Definitely deserving of credit for anything good from this project. Thank you!
- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!

## Contact 💌

For questions or feedback, please raise an issue or reach out to @beveradb ([Andrew Beveridge](mailto:andrew@beveridge.uk)) directly.
