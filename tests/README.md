# Audio Separator Tests

This directory contains tests for the audio-separator project.

## Audio Validation

The integration tests now include validation of output audio files by comparing waveform and spectrogram images with reference images. This helps ensure that the audio separation results remain consistent across different runs and code changes.

### How It Works

1. Reference waveform and spectrogram images are generated from expected output files
2. During test execution, the same images are generated for the actual output files
3. The images are compared to ensure they are similar, with a configurable minimum similarity threshold
4. If the images differ significantly, the test fails, indicating a change in the audio output

### Similarity Threshold

The comparison uses a minimum similarity threshold (0.0-1.0):
- **Higher values** (closer to 1.0) require images to be **more similar**
- **Lower values** (closer to 0.0) are **more permissive**
- A value of 0.99 requires 99% similarity between images
- A value of 0.0 would consider any images to match

The default threshold is set to 0.999, which is quite strict. However, model-specific thresholds can be set to accommodate different models' behavior.

#### Model-Specific Thresholds

Some models inherently produce slightly different outputs on different runs, even with the same input. To accommodate these models, you can set model-specific thresholds in the `MODEL_SIMILARITY_THRESHOLDS` dictionary:

```python
MODEL_SIMILARITY_THRESHOLDS = {
    "htdemucs_6s.yaml": 0.995,  # Demucs models need a lower threshold
    # Add other models that need custom thresholds here
}
```

This allows you to maintain a high threshold for most models while being more flexible with models that naturally exhibit more variation.

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
5. If necessary, add a custom similarity threshold for the new model in `MODEL_SIMILARITY_THRESHOLDS`

### Debugging

To see detailed validation results, run pytest with the `-sv` flag:

```bash
pytest tests/integration/test_cli_integration.py -sv
```

This will show the similarity scores for each comparison and whether they passed or failed.

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