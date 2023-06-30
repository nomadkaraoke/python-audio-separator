# Audio Separator

[![PyPI version](https://badge.fury.io/py/audio-separator.svg)](https://badge.fury.io/py/audio-separator)

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
audio-separator [audio_file] --model_name [model_name]
    
    audio_file: The path to the WAV audio file to be separated.
    model_name: (Optional) The name of the model to use for separation. Default: UVR_MDXNET_KARA_2
    model_file_dir: (Optional) Directory to cache model files in. Default: /tmp/audio-separator-models/
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

Python <= 3.10 (one of the dependencies doesn't like 3.11 yet)

Libraries: onnx, onnxruntime, numpy, soundfile, librosa, torch, wget, six

## License

This project is licensed under the MIT [License](LICENSE).

- **Please Note:** If you choose to integrate this project into some other project using the default model or any other model trained as part of the [UVR](https://github.com/Anjok07/ultimatevocalremovergui) project, please honor the MIT license by providing credit to UVR and its developers!

## Credits

- [Anjok07](https://github.com/Anjok07) - Author of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui), which almost all of the code in this repo was copied from! Definitely deserving of credit for anything good from this project. Thank you!
- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!

## Contributing

Contributions are very much welcome! Please fork the repository and submit a pull request with your changes, and I'll try to review, merge and publish promptly!

- This project is 100% open-source and free for anyone to use and modify as they wish. 
- If the maintenance workload for this repo somehow becomes too much for me I'll ask for volunteers to share maintainership of the repo, though I don't think that is very likely
- Development and support for the MDX-Net separation models is part of the main [UVR project](https://github.com/Anjok07/ultimatevocalremovergui), this repo is just a CLI/Python package wrapper to simplify running those models programmatically. So, if you want to try and improve the actual models, please get involved in the UVR project and look for guidance there!
