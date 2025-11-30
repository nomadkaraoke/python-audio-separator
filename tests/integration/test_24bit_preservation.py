"""
Integration tests for 24-bit audio bit depth preservation.

Tests that 24-bit input audio files produce 24-bit output audio files
while maintaining audio quality through the separation process.
"""

import os
import subprocess
import pytest
import soundfile as sf
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import generate_reference_images, compare_images


@pytest.fixture(name="input_file_24bit")
def fixture_input_file_24bit():
    """Fixture providing the 24-bit test input audio file path."""
    return "tests/inputs/fallen24bit20s.flac"


@pytest.fixture(name="reference_dir")
def fixture_reference_dir():
    """Fixture providing the reference images directory path."""
    return "tests/inputs/reference"


@pytest.fixture(name="cleanup_output_files")
def fixture_cleanup_output_files():
    """Fixture to clean up output files before and after test."""
    output_files = []
    yield output_files
    
    # Clean up output files after test
    for file in output_files:
        if os.path.exists(file):
            print(f"Cleaning up test output file: {file}")
            os.remove(file)


def get_audio_bit_depth(file_path):
    """Get the bit depth of an audio file."""
    info = sf.info(file_path)
    subtype = info.subtype
    
    if 'PCM_16' in subtype or subtype == 'PCM_S8':
        return 16
    elif 'PCM_24' in subtype:
        return 24
    elif 'PCM_32' in subtype or 'FLOAT' in subtype or 'DOUBLE' in subtype:
        return 32
    else:
        return None


def run_separation_test_24bit(model, audio_path, expected_files):
    """Helper function to run a separation test with a 24-bit input file."""
    # Clean up any existing output files before the test
    for file in expected_files:
        if os.path.exists(file):
            print(f"Deleting existing test output file {file}")
            os.remove(file)
    
    # Verify input is 24-bit
    input_bit_depth = get_audio_bit_depth(audio_path)
    print(f"Input file bit depth: {input_bit_depth}-bit")
    assert input_bit_depth == 24, f"Input file should be 24-bit, got {input_bit_depth}-bit"
    
    # Run the CLI command
    result = subprocess.run(
        ["audio-separator", "-m", model, audio_path],
        capture_output=True,
        text=True,
        check=False
    )
    
    # Check that the command completed successfully
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"
    
    # Check that the output files were created and are 24-bit
    for file in expected_files:
        assert os.path.exists(file), f"Output file {file} was not created"
        assert os.path.getsize(file) > 0, f"Output file {file} is empty"
        
        # Verify output is also 24-bit
        output_bit_depth = get_audio_bit_depth(file)
        print(f"Output file {file} bit depth: {output_bit_depth}-bit")
        assert output_bit_depth == 24, f"Output file should be 24-bit to match input, got {output_bit_depth}-bit"
    
    return result


def validate_audio_output(output_file, reference_dir, waveform_threshold=0.999, spectrogram_threshold=None):
    """Validate an audio output file by comparing its waveform and spectrogram with reference images."""
    if spectrogram_threshold is None:
        spectrogram_threshold = waveform_threshold
    
    # Create temporary directory for generated images
    temp_dir = os.path.join(os.path.dirname(output_file), "temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate waveform and spectrogram images for the output file
    output_filename = os.path.basename(output_file)
    name_without_ext = os.path.splitext(output_filename)[0]
    
    # Generate actual images
    actual_waveform_path, actual_spectrogram_path = generate_reference_images(
        output_file, temp_dir, prefix="actual_"
    )
    
    # Path to expected reference images
    expected_waveform_path = os.path.join(
        reference_dir, f"expected_{name_without_ext}_waveform.png"
    )
    expected_spectrogram_path = os.path.join(
        reference_dir, f"expected_{name_without_ext}_spectrogram.png"
    )
    
    # Check if reference images exist
    if not os.path.exists(expected_waveform_path) or not os.path.exists(expected_spectrogram_path):
        print(f"Warning: Reference images not found for {output_file}")
        print(f"Expected: {expected_waveform_path} and {expected_spectrogram_path}")
        print(f"Run generate_reference_images_24bit.py to create them")
        return False, False
    
    # Compare waveform images
    waveform_similarity, waveform_match = compare_images(
        expected_waveform_path, actual_waveform_path,
        min_similarity_threshold=waveform_threshold
    )
    
    # Compare spectrogram images
    spectrogram_similarity, spectrogram_match = compare_images(
        expected_spectrogram_path, actual_spectrogram_path,
        min_similarity_threshold=spectrogram_threshold
    )
    
    print(f"Validation results for {output_file}:")
    print(f"  Waveform similarity: {waveform_similarity:.4f} "
          f"(match: {waveform_match}, threshold: {waveform_threshold:.2f})")
    print(f"  Spectrogram similarity: {spectrogram_similarity:.4f} "
          f"(match: {spectrogram_match}, threshold: {spectrogram_threshold:.2f})")
    
    return waveform_match, spectrogram_match


# Default similarity threshold for 24-bit audio tests
DEFAULT_SIMILARITY_THRESHOLDS_24BIT = (0.90, 0.80)  # (waveform, spectrogram)

# Model-specific similarity thresholds for 24-bit tests
MODEL_SIMILARITY_THRESHOLDS_24BIT = {
    # Format: (waveform_threshold, spectrogram_threshold)
}


# Parameterized test for multiple models with 24-bit audio
MODEL_PARAMS_24BIT = [
    # (model_filename, expected_output_filenames)
    (
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        [
            "fallen24bit20s_(Instrumental)_model_bs_roformer_ep_317_sdr_12.flac",
            "fallen24bit20s_(Vocals)_model_bs_roformer_ep_317_sdr_12.flac",
        ]
    ),
    (
        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        [
            "fallen24bit20s_(Instrumental)_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.flac",
            "fallen24bit20s_(Vocals)_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.flac",
        ]
    ),
    (
        "MGM_MAIN_v4.pth",
        [
            "fallen24bit20s_(Instrumental)_MGM_MAIN_v4.flac",
            "fallen24bit20s_(Vocals)_MGM_MAIN_v4.flac",
        ]
    ),
]


@pytest.mark.parametrize("model,expected_files", MODEL_PARAMS_24BIT)
def test_24bit_model_separation(model, expected_files, input_file_24bit, reference_dir, cleanup_output_files):
    """Test that 24-bit input audio produces 24-bit output audio with correct content."""
    print(f"\n{'='*60}")
    print(f"Testing 24-bit preservation with model: {model}")
    print(f"{'='*60}")
    
    # Add files to the cleanup list
    cleanup_output_files.extend(expected_files)
    
    # Run the test (includes bit depth validation)
    run_separation_test_24bit(model, input_file_24bit, expected_files)
    
    # Validate the output audio quality
    print(f"\nValidating output audio quality for model {model}...")
    
    # Get model-specific similarity threshold or use default
    threshold = MODEL_SIMILARITY_THRESHOLDS_24BIT.get(
        model, DEFAULT_SIMILARITY_THRESHOLDS_24BIT
    )
    waveform_threshold, spectrogram_threshold = threshold
    
    print(f"Using thresholds - waveform: {waveform_threshold}, "
          f"spectrogram: {spectrogram_threshold}")
    
    for output_file in expected_files:
        # Skip validation if reference images are not required
        if os.environ.get("SKIP_AUDIO_VALIDATION") == "1":
            print(f"Skipping audio validation for {output_file} (SKIP_AUDIO_VALIDATION=1)")
            continue
        
        waveform_match, spectrogram_match = validate_audio_output(
            output_file, reference_dir,
            waveform_threshold=waveform_threshold,
            spectrogram_threshold=spectrogram_threshold
        )
        
        # Assert that the output matches the reference
        assert waveform_match, f"Waveform for {output_file} does not match the reference"
        assert spectrogram_match, f"Spectrogram for {output_file} does not match the reference"
    
    print(f"✅ Test passed for {model}: 24-bit preservation and audio quality verified")

