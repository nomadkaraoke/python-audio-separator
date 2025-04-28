# Audio Separator Tests

This directory contains tests for the audio-separator project.

## Audio Validation

The integration tests now include validation of output audio files by comparing waveform and spectrogram images with reference images. This helps ensure that the audio separation results remain consistent across different runs and code changes.

### How It Works

1. Reference waveform and spectrogram images are generated from expected output files
2. During test execution, the same images are generated for the actual output files
3. The images are compared to ensure they are similar, with a configurable threshold
4. If the images differ significantly, the test fails, indicating a change in the audio output

### Generating Reference Images

To generate or update the reference images, use the script provided:

```bash
python tests/integration/generate_reference_images.py
```

This script will create waveform and spectrogram images for all expected output files and store them in the `tests/inputs/reference` directory.

### Skipping Validation

If you need to skip the audio validation (e.g., when you're intentionally changing the output), you can set the environment variable `SKIP_AUDIO_VALIDATION=1`:

```bash
SKIP_AUDIO_VALIDATION=1 pytest tests/integration/test_cli_integration.py
```

### Adding New Models

When adding a new model to the tests:

1. Add the model and its expected output files to the `MODEL_PARAMS` list in `test_cli_integration.py`
2. Run the integration test to generate the output files
3. Run the `generate_reference_images.py` script to create the reference images
4. Run the tests again to validate the output files

## Running Tests

To run all tests:

```bash
pytest
```

To run specific tests:

```bash
pytest tests/unit/
pytest tests/integration/
```

To run with coverage:

```bash
pytest --cov=audio_separator
``` 