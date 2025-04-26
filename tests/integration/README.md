# Integration Tests

These tests verify the end-to-end functionality of the audio-separator CLI.

## Running the tests

To run the integration tests, use:

```bash
pytest tests/integration
```

To run a specific model test, you can use pytest's parameter selection:

```bash
# Run only the kuielab_b_vocals.onnx test
pytest tests/integration/test_cli_integration.py::test_model_separation[kuielab_b_vocals.onnx-expected_files0]

# Run only the MGM_MAIN_v4.pth test
pytest tests/integration/test_cli_integration.py::test_model_separation[MGM_MAIN_v4.pth-expected_files1]
```

## Adding New Model Tests

To add a new model test, simply add a new entry to the `MODEL_PARAMS` list in the test file:

```python
(
    "new_model_filename.onnx",
    ["mardy20s_(Instrumental)_new_model_filename.flac", "mardy20s_(Vocals)_new_model_filename.flac"]
),
```

No additional test functions are needed.

## Notes

- These tests use actual audio files and models, and will run the full audio separation process.
- Tests may take longer to run than unit tests, as they perform actual audio processing.
- The model files will be automatically downloaded if they don't exist locally.
- The test requires the test audio file at `tests/inputs/mardy20s.flac` to exist. 