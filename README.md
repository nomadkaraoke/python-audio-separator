# Audio Separator 🎶

[![PyPI version](https://badge.fury.io/py/audio-separator.svg)](https://badge.fury.io/py/audio-separator)

Summary: Easy to use vocal separation on CLI or as a python package, using the amazing MDX-Net models from UVR trained by @Anjok07

Audio Separator is a Python package that allows you to separate an audio file into two stems, primary and secondary, using a model in the ONNX format trained by @Anjok07 for use with UVR (https://github.com/Anjok07/ultimatevocalremovergui).

The primary stem typically contains the instrumental part of the audio, while the secondary stem contains the vocals, but in some models this is reversed.

## Features

- Separate audio into instrumental and vocal stems.
- Supports all common audio formats (WAV, MP3, FLAC, M4A, etc.)
- Ability to specify a pre-trained deep learning model in ONNX format.
- CLI support for easy use in scripts and batch processing.
- Python API for integration into other projects.

## Installation 🛠️

You can install Audio Separator using pip:

`pip install audio-separator`

### Extra installation steps for use with a GPU

Unfortunately the way Torch and ONNX Runtime are published means the correct platform-specific dependencies for CUDA use don't get installed by the package published to PyPI with Poetry.

As such, if you want to use audio-separator with a CUDA-capable Nvidia GPU, you need to reinstall them directly, allowing pip to calculate the right versions for your platform:

- `pip uninstall torch onnxruntime`
- `pip cache purge`
- `pip install torch "optimum[onnxruntime-gpu]"`

This should get you set up to run audio-separator with CUDA acceleration, using the `--use_cuda` argument.

> Note: if anyone has a way to make this cleaner so we can support both CPU and CUDA transcodes without separate installation processes, please let me know or submit a PR!

## Usage 🚀

### Command Line Interface (CLI)

You can use Audio Separator via the command line:

```sh
audio-separator [audio_file] --model_name [model_name]
    
    audio_file: The path to the audio file to be separated. Supports all common formats (WAV, MP3, FLAC, M4A, etc.)
    log_level: (Optional) Logging level, e.g. info, debug, warning. Default: INFO
    log_formatter: (Optional) The log format. Default: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    model_name: (Optional) The name of the model to use for separation. Default: UVR_MDXNET_KARA_2
    model_file_dir: (Optional) Directory to cache model files in. Default: /tmp/audio-separator-models/
    output_dir: (Optional) The directory where the separated files will be saved. If not specified, outputs to current dir.
    use_cuda: (Optional) Flag to use Nvidia GPU via CUDA for separation if available. Default: False
    denoise_enabled: (Optional) Flag to enable or disable denoising as part of the separation process. Default: True
    normalization_enabled: (Optional) Flag to enable or disable normalization as part of the separation process. Default: False
    output_format: (Optional) Format to encode output files, any common format (WAV, MP3, FLAC, M4A, etc.). Default: WAV
    single_stem: (Optional) Output only single stem, either instrumental or vocals. Example: --single_stem=instrumental
```

Example:

```
audio-separator /path/to/your/audio.wav --model_name UVR_MDXNET_KARA_2
```

This command will process the file and generate two new files in the current directory, one for each stem.

### As a Dependency in a Python Project

You can also use Audio Separator in your Python project. Here's how you can use it:

```
from audio_separator import Separator

# Initialize the Separator with the audio file and model name
separator = Separator('/path/to/your/audio.m4a', model_name='UVR_MDXNET_KARA_2')

# Perform the separation
primary_stem_path, secondary_stem_path = separator.separate()

print(f'Primary stem saved at {primary_stem_path}')
print(f'Secondary stem saved at {secondary_stem_path}')
```

## Parameters for the Separator class

- audio_file: The path to the audio file to be separated. Supports all common formats (WAV, MP3, FLAC, M4A, etc.)
- log_level: (Optional) Logging level, e.g. info, debug, warning. Default: INFO
- log_formatter: (Optional) The log format. Default: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
- model_name: (Optional) The name of the model to use for separation. Defaults to 'UVR_MDXNET_KARA_2', a very powerful model for Karaoke instrumental tracks.
- model_file_dir: (Optional) Directory to cache model files in. Default: /tmp/audio-separator-models/
- output_dir: (Optional) Directory where the separated files will be saved. If not specified, outputs to current dir.
- use_cuda: (Optional) Flag to use Nvidia GPU via CUDA for separation if available. Default: False
- denoise_enabled: (Optional) Flag to enable or disable denoising as part of the separation process. Default: True
- normalization_enabled: (Optional) Flag to enable or disable normalization as part of the separation process. Default: False
- output_format: (Optional) Format to encode output files, any common format (WAV, MP3, FLAC, M4A, etc.). Default: WAV
- output_single_stem: (Optional) Output only single stem, either instrumental or vocals.

## Requirements 📋

Python >= 3.9

Libraries: onnx, onnxruntime, numpy, soundfile, librosa, torch, wget, six

## Developing Locally

This project uses Poetry for dependency management and packaging. Follow these steps to setup a local development environment:

### Prerequisites

- Make sure you have Python 3.9 or newer installed on your machine.
- Install Poetry by following the installation guide here.

### Clone the Repository

Clone the repository to your local machine:

```
git clone https://github.com/YOUR_USERNAME/audio-separator.git
cd audio-separator
```

Replace YOUR_USERNAME with your GitHub username if you've forked the repository, or use the main repository URL if you have the permissions.

### Install Dependencies

Run the following command to install the project dependencies:

```
poetry install
```

### Activate the Virtual Environment

To activate the virtual environment, use the following command:

```
poetry shell
```

### Running the Command-Line Interface Locally

You can run the CLI command directly within the virtual environment. For example:

```
audio-separator path/to/your/audio-file.wav
```

### Deactivate the Virtual Environment

Once you are done with your development work, you can exit the virtual environment by simply typing:

```
exit
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
