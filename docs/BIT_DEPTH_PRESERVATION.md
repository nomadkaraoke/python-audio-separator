# Bit Depth Preservation Implementation

## Summary

This implementation ensures that the output audio files from audio-separator preserve the bit depth of the input audio files, preventing quality loss when processing 24-bit or 32-bit audio files.

## Changes Made

### 1. Added `soundfile` to Dependencies (`pyproject.toml`)
- Added `soundfile >= 0.12` as a dependency to enable reading audio file metadata

### 2. Modified `CommonSeparator` Class (`audio_separator/separator/common_separator.py`)

#### Added Bit Depth Tracking
- Added `self.input_bit_depth` attribute to store the detected bit depth (16, 24, or 32)
- Added `self.input_subtype` attribute to store the audio file subtype

#### Modified `prepare_mix()` Method
- Added code to read audio file metadata using `soundfile.info()` before loading with librosa
- Detects and stores the input audio file's bit depth based on the subtype:
  - PCM_16, PCM_S8 → 16-bit
  - PCM_24 → 24-bit
  - PCM_32, FLOAT, DOUBLE → 32-bit
- Defaults to 16-bit for numpy arrays or unknown formats
- Logs the detected bit depth for debugging

#### Modified `write_audio_pydub()` Method
- Determines output bit depth from `self.input_bit_depth`
- Scales audio data appropriately for each bit depth:
  - 16-bit: scale by 32767, use int16, sample_width=2
  - 24-bit: scale by 8388607, use int32, sample_width=3
  - 32-bit: scale by 2147483647, use int32, sample_width=4
- Passes appropriate ffmpeg codec parameters for WAV files:
  - 16-bit: `pcm_s16le`
  - 24-bit: `pcm_s24le`
  - 32-bit: `pcm_s32le`

#### Modified `write_audio_soundfile()` Method
- Determines output subtype from `self.input_subtype` or `self.input_bit_depth`
- Passes the subtype to `sf.write()` to preserve bit depth
- Removed manual interleaving (soundfile handles multi-channel properly)

## Tests Created

### Unit Tests

#### `tests/unit/test_bit_depth_detection.py`
Tests bit depth detection logic:
- `test_16bit_detection`: Verifies 16-bit files are correctly detected
- `test_24bit_detection`: Verifies 24-bit files are correctly detected
- `test_32bit_detection`: Verifies 32-bit files are correctly detected
- `test_numpy_array_input_defaults_to_16bit`: Verifies numpy arrays default to 16-bit
- `test_bit_depth_preserved_across_multiple_files`: Verifies bit depth updates between files

#### `tests/unit/test_bit_depth_writing.py`
Tests bit depth preservation in write functions:
- `test_write_16bit_with_pydub`: Tests 16-bit output with pydub backend
- `test_write_24bit_with_pydub`: Tests 24-bit output with pydub backend
- `test_write_32bit_with_pydub`: Tests 32-bit output with pydub backend
- `test_write_24bit_with_soundfile`: Tests 24-bit output with soundfile backend
- `test_write_16bit_with_soundfile`: Tests 16-bit output with soundfile backend

### Integration Tests

#### `tests/integration/test_bit_depth_e2e.py`
End-to-end tests with full separation workflow:
- `test_e2e_24bit_preservation`: Full test with 24-bit input
- `test_e2e_16bit_preservation`: Full test with 16-bit input

#### `tests/integration/test_bit_depth_preservation.py`
Comprehensive integration tests (skipped if package not installed):
- Tests for 16-bit, 24-bit, and 32-bit preservation
- Tests with FLAC output format
- Tests with soundfile backend
- Tests with multiple files of different bit depths

## Test Results

All unit tests pass successfully:
- 5/5 bit depth detection tests ✅
- 5/5 bit depth writing tests ✅
- 10/10 total unit tests ✅

## Behavior

### Before
- All output files were written as 16-bit PCM, regardless of input bit depth
- This resulted in quality loss when processing 24-bit or 32-bit audio files

### After
- Output bit depth matches input bit depth automatically
- 16-bit input → 16-bit output
- 24-bit input → 24-bit output
- 32-bit input → 32-bit output
- No quality loss when processing high-quality audio files
- Backward compatible - 16-bit inputs still produce 16-bit outputs

## Supported Formats

Bit depth preservation works with:
- **WAV files**: Full support for 16, 24, and 32-bit
- **FLAC files**: Full support for 16, 24, and 32-bit
- **Other formats**: Depends on ffmpeg support through pydub

Both output backends are supported:
- **pydub/ffmpeg** (default): Uses codec parameters to enforce bit depth
- **soundfile**: Uses subtype parameter to enforce bit depth

## Notes

- The implementation is backward compatible - no changes to the API are required
- Bit depth is detected automatically from input files
- Logs provide visibility into detected and output bit depths
- Falls back to 16-bit for unknown formats or errors
- Works correctly when processing multiple files with different bit depths

