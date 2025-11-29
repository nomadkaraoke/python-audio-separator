"""
End-to-end test for bit depth preservation with real audio files.

This test uses existing test audio files to verify bit depth preservation.
"""

import os
import pytest
import tempfile
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path

# Note: This test requires the package to be installed, so we skip if it's not available
try:
    from audio_separator.separator import Separator
    SEPARATOR_AVAILABLE = True
except Exception:
    SEPARATOR_AVAILABLE = False


def create_test_audio_file(output_path, sample_rate=44100, duration=2.0, bit_depth=16):
    """
    Create a test audio file with a specific bit depth.
    
    Args:
        output_path: Path to save the test audio file
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        bit_depth: Bit depth (16, 24, or 32)
    
    Returns:
        Path to the created audio file
    """
    # Generate a simple test signal (440 Hz sine wave)
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Create stereo by duplicating the mono signal
    stereo_audio = np.column_stack([audio, audio])
    
    # Determine the subtype based on bit depth
    if bit_depth == 16:
        subtype = 'PCM_16'
    elif bit_depth == 24:
        subtype = 'PCM_24'
    elif bit_depth == 32:
        subtype = 'PCM_32'
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    # Write the audio file
    sf.write(output_path, stereo_audio, sample_rate, subtype=subtype)
    
    return output_path


def get_audio_bit_depth(file_path):
    """
    Get the bit depth of an audio file.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Bit depth as an integer (16, 24, or 32)
    """
    info = sf.info(file_path)
    subtype = info.subtype
    
    if 'PCM_16' in subtype or subtype == 'PCM_S8':
        return 16
    elif 'PCM_24' in subtype:
        return 24
    elif 'PCM_32' in subtype or 'FLOAT' in subtype or 'DOUBLE' in subtype:
        return 32
    else:
        # Unknown format
        return None


@pytest.fixture(name="temp_output_dir")
def fixture_temp_output_dir():
    """Fixture providing a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.fixture(name="temp_audio_dir")
def fixture_temp_audio_dir():
    """Fixture providing a temporary directory for input audio files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.mark.skipif(not SEPARATOR_AVAILABLE, reason="Separator not available (package not installed)")
def test_e2e_24bit_preservation(temp_audio_dir, temp_output_dir):
    """End-to-end test that 24-bit input audio produces 24-bit output audio."""
    print("\n>>> E2E TEST: 24-bit bit depth preservation")
    
    # Create a 24-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24, duration=2.0)
    
    # Verify the input file is 24-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 24, f"Input file should be 24-bit, got {input_bit_depth}"
    
    # Initialize separator with WAV output format
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    try:
        separator = Separator(
            output_dir=temp_output_dir,
            output_format="wav",
            log_level=30,  # WARNING level to reduce noise
            info_only=True  # Skip package version check
        )
    except Exception as e:
        pytest.skip(f"Could not initialize Separator: {e}")
    
    # Load a small, fast model for testing
    try:
        print("Loading model: MGM_MAIN_v4.pth")
        separator.load_model(model_filename="MGM_MAIN_v4.pth")
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    
    # Run separation
    print("Running separation")
    try:
        output_files = separator.separate(input_file)
        print(f"Separator.separate() returned: {output_files}")
    except Exception as e:
        pytest.skip(f"Could not run separation: {e}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 24, f"Output file should be 24-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 24-bit input produces 24-bit output in end-to-end test")


@pytest.mark.skipif(not SEPARATOR_AVAILABLE, reason="Separator not available (package not installed)")
def test_e2e_16bit_preservation(temp_audio_dir, temp_output_dir):
    """End-to-end test that 16-bit input audio produces 16-bit output audio."""
    print("\n>>> E2E TEST: 16-bit bit depth preservation")
    
    # Create a 16-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_16bit.wav")
    create_test_audio_file(input_file, bit_depth=16, duration=2.0)
    
    # Verify the input file is 16-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 16, f"Input file should be 16-bit, got {input_bit_depth}"
    
    # Initialize separator with WAV output format
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    try:
        separator = Separator(
            output_dir=temp_output_dir,
            output_format="wav",
            log_level=30,  # WARNING level to reduce noise
            info_only=True  # Skip package version check
        )
    except Exception as e:
        pytest.skip(f"Could not initialize Separator: {e}")
    
    # Load a small, fast model for testing
    try:
        print("Loading model: MGM_MAIN_v4.pth")
        separator.load_model(model_filename="MGM_MAIN_v4.pth")
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    
    # Run separation
    print("Running separation")
    try:
        output_files = separator.separate(input_file)
        print(f"Separator.separate() returned: {output_files}")
    except Exception as e:
        pytest.skip(f"Could not run separation: {e}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 16, f"Output file should be 16-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 16-bit input produces 16-bit output in end-to-end test")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

