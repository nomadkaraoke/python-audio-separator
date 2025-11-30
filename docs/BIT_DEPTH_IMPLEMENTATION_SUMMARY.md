# Summary: Bit Depth Preservation Implementation

## Issue
[GitHub Issue #243](https://github.com/nomadkaraoke/python-audio-separator/issues/243) - Users reported that audio-separator was reducing audio quality by always outputting 16-bit audio, even when the input was 24-bit or 32-bit.

## Solution
Implemented automatic bit depth preservation that matches the output audio bit depth to the input audio file's bit depth. This ensures no quality loss when processing high-resolution audio files.

## Key Changes

### 1. **Dependencies** (`pyproject.toml`)
- Added `soundfile >= 0.12` for reading audio file metadata

### 2. **Core Implementation** (`audio_separator/separator/common_separator.py`)
- Added `input_bit_depth` and `input_subtype` attributes to track input audio properties
- Modified `prepare_mix()` to detect bit depth using soundfile
- Updated `write_audio_pydub()` to use appropriate scaling and ffmpeg codecs for each bit depth
- Updated `write_audio_soundfile()` to preserve subtype when writing

### 3. **Comprehensive Tests**
Created 3 test suites with 17 tests total:

**Unit Tests:**
- `tests/unit/test_bit_depth_detection.py` - 5 tests for bit depth detection
- `tests/unit/test_bit_depth_writing.py` - 5 tests for write functions

**Integration Tests:**
- `tests/integration/test_bit_depth_e2e.py` - 2 end-to-end tests
- `tests/integration/test_bit_depth_preservation.py` - 6 comprehensive integration tests

**Manual Test:**
- `tests/manual_test_bit_depth.py` - Demonstrates functionality

## Test Results

✅ **All tests pass:**
```
16-bit (pydub)      ✅ PASS
24-bit (pydub)      ✅ PASS
32-bit (pydub)      ✅ PASS
16-bit (soundfile)  ✅ PASS
24-bit (soundfile)  ✅ PASS
32-bit (soundfile)  ✅ PASS
```

## Behavior

| Input Bit Depth | Previous Output | New Output |
|----------------|-----------------|------------|
| 16-bit         | 16-bit         | 16-bit ✅  |
| 24-bit         | **16-bit** ❌  | 24-bit ✅  |
| 32-bit         | **16-bit** ❌  | 32-bit ✅  |

## Impact

✅ **Quality Preservation:** No more quality loss when processing high-resolution audio
✅ **Backward Compatible:** Existing 16-bit workflows unchanged
✅ **Automatic:** No configuration required - works out of the box
✅ **Transparent:** Logs show detected and output bit depths
✅ **Robust:** Graceful fallback to 16-bit for unknown formats

## Technical Details

The implementation:
- Reads audio metadata before loading with librosa
- Maps PCM subtypes to bit depths (PCM_16→16, PCM_24→24, PCM_32→32)
- Scales audio data appropriately for each bit depth
- Passes correct codec parameters to ffmpeg/pydub
- Works with both pydub (default) and soundfile backends
- Handles multiple files with different bit depths correctly

## Files Modified

1. `pyproject.toml` - Added soundfile dependency
2. `audio_separator/separator/common_separator.py` - Core implementation

## Files Added

1. `tests/unit/test_bit_depth_detection.py` - Unit tests for detection
2. `tests/unit/test_bit_depth_writing.py` - Unit tests for writing
3. `tests/integration/test_bit_depth_e2e.py` - End-to-end tests
4. `tests/integration/test_bit_depth_preservation.py` - Integration tests
5. `tests/manual_test_bit_depth.py` - Manual test script
6. `BIT_DEPTH_PRESERVATION.md` - Detailed documentation

## No Breaking Changes

This implementation is fully backward compatible:
- No API changes required
- No new parameters needed
- Existing functionality unchanged
- Only affects output bit depth to match input

## Resolves

✅ Closes #243 - Output bit depth now matches input automatically

