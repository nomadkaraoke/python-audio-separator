# Integration Tests

These tests verify the end-to-end functionality of the audio-separator CLI.

## Test Files

### 16-bit Audio Tests
- Input: `tests/inputs/mardy20s.flac` (16-bit)
- Tests: `test_cli_integration.py`
- Validates separation with various models using 16-bit audio

### 24-bit Audio Tests
- Input: `tests/inputs/fallen24bit20s.flac` (24-bit)
- Tests: `test_24bit_preservation.py`
- Validates that 24-bit input produces 24-bit output
- Ensures audio quality is preserved during bit depth preservation

## Running the tests

To run all integration tests:

```bash
pytest tests/integration
```

To run only 16-bit tests:

```bash
pytest tests/integration/test_cli_integration.py
```

To run only 24-bit preservation tests:

```bash
pytest tests/integration/test_24bit_preservation.py
```

To run a specific model test, use pytest's parameter selection:

```bash
# Run only the kuielab_b_vocals.onnx test (16-bit)
pytest tests/integration/test_cli_integration.py::test_model_separation[kuielab_b_vocals.onnx-expected_files0]

# Run only the BS-Roformer test with 24-bit audio
pytest tests/integration/test_24bit_preservation.py::test_24bit_model_separation[model_bs_roformer_ep_317_sdr_12.9755.ckpt-expected_files0]
```

## Adding New Model Tests

### For 16-bit tests
Add a new entry to the `MODEL_PARAMS` list in `test_cli_integration.py`:

```python
(
    "new_model_filename.onnx",
    ["mardy20s_(Instrumental)_new_model_filename.flac", "mardy20s_(Vocals)_new_model_filename.flac"]
),
```

### For 24-bit tests
Add a new entry to the `MODEL_PARAMS_24BIT` list in `test_24bit_preservation.py`:

```python
(
    "new_model_filename.onnx",
    ["fallen24bit20s_(Instrumental)_new_model_filename.flac", "fallen24bit20s_(Vocals)_new_model_filename.flac"]
),
```

## Generating Reference Images

### For 16-bit tests
```bash
python tests/integration/generate_reference_images.py
```

### For 24-bit tests
1. First, generate the output files by running audio-separator:
   ```bash
   audio-separator -m model_name.ckpt tests/inputs/fallen24bit20s.flac
   ```
2. Move the output files to `tests/inputs/`
3. Generate reference images:
   ```bash
   python tests/integration/generate_reference_images_24bit.py
   ```

## Notes

- These tests use actual audio files and models, and will run the full audio separation process.
- Tests may take longer to run than unit tests, as they perform actual audio processing.
- The model files will be automatically downloaded if they don't exist locally.
- The 16-bit test requires the test audio file at `tests/inputs/mardy20s.flac` to exist.
- The 24-bit test requires the test audio file at `tests/inputs/fallen24bit20s.flac` to exist.
- Reference images must be generated before running the tests for the first time. 