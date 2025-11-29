"""
Integration tests for bit depth preservation.

Tests that the output audio files preserve the bit depth of the input audio files.
"""

import os
import pytest
import tempfile
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path

from audio_separator.separator import Separator


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


def test_16bit_preservation(temp_audio_dir, temp_output_dir):
    """Test that 16-bit input audio produces 16-bit output audio."""
    print("\n>>> TEST: 16-bit bit depth preservation")
    
    # Create a 16-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_16bit.wav")
    create_test_audio_file(input_file, bit_depth=16)
    
    # Verify the input file is 16-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 16, f"Input file should be 16-bit, got {input_bit_depth}"
    
    # Initialize separator with WAV output format
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="wav",
        log_level=20  # INFO level
    )
    
    # Load a small, fast model for testing
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")
    
    # Run separation
    print("Running separation")
    output_files = separator.separate(input_file)
    print(f"Separator.separate() returned: {output_files}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 16, f"Output file should be 16-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 16-bit input produces 16-bit output")


def test_24bit_preservation(temp_audio_dir, temp_output_dir):
    """Test that 24-bit input audio produces 24-bit output audio."""
    print("\n>>> TEST: 24-bit bit depth preservation")
    
    # Create a 24-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24)
    
    # Verify the input file is 24-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 24, f"Input file should be 24-bit, got {input_bit_depth}"
    
    # Initialize separator with WAV output format
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="wav",
        log_level=20  # INFO level
    )
    
    # Load a small, fast model for testing
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")
    
    # Run separation
    print("Running separation")
    output_files = separator.separate(input_file)
    print(f"Separator.separate() returned: {output_files}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 24, f"Output file should be 24-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 24-bit input produces 24-bit output")


def test_32bit_preservation(temp_audio_dir, temp_output_dir):
    """Test that 32-bit input audio produces 32-bit output audio."""
    print("\n>>> TEST: 32-bit bit depth preservation")
    
    # Create a 32-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_32bit.wav")
    create_test_audio_file(input_file, bit_depth=32)
    
    # Verify the input file is 32-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 32, f"Input file should be 32-bit, got {input_bit_depth}"
    
    # Initialize separator with WAV output format
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="wav",
        log_level=20  # INFO level
    )
    
    # Load a small, fast model for testing
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")
    
    # Run separation
    print("Running separation")
    output_files = separator.separate(input_file)
    print(f"Separator.separate() returned: {output_files}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 32, f"Output file should be 32-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 32-bit input produces 32-bit output")


def test_bit_depth_with_flac_format(temp_audio_dir, temp_output_dir):
    """Test that bit depth is preserved with FLAC output format."""
    print("\n>>> TEST: Bit depth preservation with FLAC output format")
    
    # Create a 24-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24)
    
    # Verify the input file is 24-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 24, f"Input file should be 24-bit, got {input_bit_depth}"
    
    # Initialize separator with FLAC output format
    print(f"Creating Separator with output_dir: {temp_output_dir}, output_format: flac")
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="flac",
        log_level=20  # INFO level
    )
    
    # Load a small, fast model for testing
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")
    
    # Run separation
    print("Running separation")
    output_files = separator.separate(input_file)
    print(f"Separator.separate() returned: {output_files}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 24, f"Output file should be 24-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 24-bit input produces 24-bit FLAC output")


def test_bit_depth_with_soundfile_backend(temp_audio_dir, temp_output_dir):
    """Test that bit depth is preserved when using soundfile backend."""
    print("\n>>> TEST: Bit depth preservation with soundfile backend")
    
    # Create a 24-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24)
    
    # Verify the input file is 24-bit
    input_bit_depth = get_audio_bit_depth(input_file)
    print(f"Input file bit depth: {input_bit_depth}")
    assert input_bit_depth == 24, f"Input file should be 24-bit, got {input_bit_depth}"
    
    # Initialize separator with soundfile backend
    print(f"Creating Separator with output_dir: {temp_output_dir}, use_soundfile: True")
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="wav",
        use_soundfile=True,  # Use soundfile backend instead of pydub
        log_level=20  # INFO level
    )
    
    # Load a small, fast model for testing
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")
    
    # Run separation
    print("Running separation")
    output_files = separator.separate(input_file)
    print(f"Separator.separate() returned: {output_files}")
    
    # Check that output files exist
    assert len(output_files) > 0, "No output files were created"
    
    # Check the bit depth of each output file
    for output_file in output_files:
        full_output_path = os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_output_path), f"Output file doesn't exist: {full_output_path}"
        
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        
        assert output_bit_depth == 24, f"Output file should be 24-bit to match input, got {output_bit_depth}"
    
    print("✅ Test passed: 24-bit input produces 24-bit output with soundfile backend")


def test_multiple_files_different_bit_depths(temp_audio_dir, temp_output_dir):
    """Test that bit depth is preserved when processing multiple files with different bit depths."""
    print("\n>>> TEST: Processing multiple files with different bit depths")
    
    # Create test audio files with different bit depths
    input_16bit = os.path.join(temp_audio_dir, "test_16bit.wav")
    input_24bit = os.path.join(temp_audio_dir, "test_24bit.wav")
    
    create_test_audio_file(input_16bit, bit_depth=16)
    create_test_audio_file(input_24bit, bit_depth=24)
    
    # Verify the input files have correct bit depths
    assert get_audio_bit_depth(input_16bit) == 16
    assert get_audio_bit_depth(input_24bit) == 24
    
    # Initialize separator
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="wav",
        log_level=20  # INFO level
    )
    
    # Load model
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")
    
    # Process 16-bit file
    print("\nProcessing 16-bit file")
    output_files_16 = separator.separate(input_16bit)
    
    # Check 16-bit outputs
    for output_file in output_files_16:
        full_output_path = os.path.join(temp_output_dir, output_file)
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        assert output_bit_depth == 16, f"Output file should be 16-bit, got {output_bit_depth}"
    
    # Process 24-bit file
    print("\nProcessing 24-bit file")
    output_files_24 = separator.separate(input_24bit)
    
    # Check 24-bit outputs
    for output_file in output_files_24:
        full_output_path = os.path.join(temp_output_dir, output_file)
        output_bit_depth = get_audio_bit_depth(full_output_path)
        print(f"Output file {output_file} bit depth: {output_bit_depth}")
        assert output_bit_depth == 24, f"Output file should be 24-bit, got {output_bit_depth}"
    
    print("✅ Test passed: Multiple files with different bit depths processed correctly")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

