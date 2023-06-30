# Audio Separator

Summary: Easy to use vocal separation on CLI or as a python package, using the amazing MDX-Net models from UVR trained by @Anjok07

Audio Separator is a Python package that allows you to separate an audio file into two stems, primary and secondary, using a model in the ONNX format trained by @Anjok07 for use with UVR (https://github.com/Anjok07/ultimatevocalremovergui).

The primary stem typically contains the instrumental part of the audio, while the secondary stem contains the vocals, but in some models this is reversed.

## Features

- Separate audio into instrumental and vocal stems.
- Ability to specify a pre-trained deep learning model.
- CLI support for easy use in scripts and batch processing.
- Python API for integration into other projects.

## Installation

You can install Audio Separator using pip:

`pip install audio-separator`


## Usage

### Command Line Interface (CLI)

You can use Audio Separator via the command line:

```sh
audio_separator [audio_file] --model_name [model_name]
audio_file: The path to the WAV audio file to be separated.
model_name: (Optional) The name of the model to use for separation.
```

Example:

audio_separator /path/to/your/audio.wav --model_name UVR_MDXNET_KARA_2

This command will process the file and generate two new files, one for each stem.

### As a Dependency in a Python Project

You can also use Audio Separator in your Python project. Here's how you can use it:

```
from audio_separator import Separator

# Initialize the Separator with the audio file and model name
separator = Separator('/path/to/your/audio.wav', model_name='UVR_MDXNET_KARA_2')

# Perform the separation
primary_stem_path, secondary_stem_path = separator.separate()

print(f'Primary stem saved at {primary_stem_path}')
print(f'Secondary stem saved at {secondary_stem_path}')
```

## Parameters for the Separator class

- audio_file: The path to the WAV audio file to be separated.
- model_name: (Optional) The name of the model to use for separation. Defaults to 'UVR_MDXNET_KARA_2', a very powerful model for Karaoke instrumental tracks.
- output_dir: (Optional) The directory where the separated files will be saved.

## Requirements
Python 3.10
Libraries: onnx, onnxruntime, numpy, soundfile, librosa, torch, wget, six

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.
