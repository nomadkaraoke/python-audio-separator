from setuptools import setup, find_packages

setup(
    name="audio-separator",
    version="0.1.2",
    packages=find_packages(),

    # Metadata
    author="Andrew Beveridge",
    author_email="andrew@beveridge.uk",
    description="Easy to use vocal separation on CLI or as a python package, using the amazing MDX-Net models from UVR trained by @Anjok07",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/karaokenerds/python-audio-separator",
    license='MIT',

    # Specify entry script
    entry_points={
        'console_scripts': [
            'audio-separator=scripts.separate:main',
        ],
    },

    # Specify python version and dependencies
    python_requires='>=3.9',
    install_requires=[
        'onnx',
        'onnxruntime==1.13.1',
        'numpy==1.23.4',
        'soundfile==0.11.0',
        'librosa==0.9.2',
        'torch==1.13.1',
        'wget==3.2',
        'six'
    ]
)
