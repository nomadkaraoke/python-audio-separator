"""
Unit tests for bit depth preservation functionality in CommonSeparator.

Tests the bit depth detection and storage logic without requiring full separation.
"""

import os
import pytest
import tempfile
import shutil
import soundfile as sf
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from audio_separator.separator.common_separator import CommonSeparator


def create_test_audio_file(output_path, sample_rate=44100, duration=0.5, bit_depth=16):
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


@pytest.fixture(name="temp_audio_dir")
def fixture_temp_audio_dir():
    """Fixture providing a temporary directory for input audio files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.fixture(name="mock_separator_config")
def fixture_mock_separator_config():
    """Fixture providing a mock separator configuration."""
    return {
        "logger": Mock(),
        "log_level": 20,
        "torch_device": Mock(),
        "torch_device_cpu": Mock(),
        "torch_device_mps": Mock(),
        "onnx_execution_provider": Mock(),
        "model_name": "test_model",
        "model_path": "/path/to/model",
        "model_data": {"training": {"instruments": ["vocals", "other"]}},
        "output_dir": None,
        "output_format": "wav",
        "output_bitrate": None,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "enable_denoise": False,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": 44100,
        "use_soundfile": False,
    }


def test_16bit_detection(temp_audio_dir, mock_separator_config):
    """Test that 16-bit audio files are correctly detected."""
    print("\n>>> TEST: 16-bit detection")
    
    # Create a 16-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_16bit.wav")
    create_test_audio_file(input_file, bit_depth=16)
    
    # Create CommonSeparator instance
    separator = CommonSeparator(mock_separator_config)
    
    # Call prepare_mix to detect bit depth
    mix = separator.prepare_mix(input_file)
    
    # Verify bit depth was detected correctly
    print(f"Detected bit depth: {separator.input_bit_depth}")
    print(f"Detected subtype: {separator.input_subtype}")
    
    assert separator.input_bit_depth == 16, f"Expected 16-bit, got {separator.input_bit_depth}"
    assert 'PCM_16' in separator.input_subtype or separator.input_subtype == 'PCM_16', f"Expected PCM_16, got {separator.input_subtype}"
    
    print("✅ Test passed: 16-bit audio correctly detected")


def test_24bit_detection(temp_audio_dir, mock_separator_config):
    """Test that 24-bit audio files are correctly detected."""
    print("\n>>> TEST: 24-bit detection")
    
    # Create a 24-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_24bit.wav")
    create_test_audio_file(input_file, bit_depth=24)
    
    # Create CommonSeparator instance
    separator = CommonSeparator(mock_separator_config)
    
    # Call prepare_mix to detect bit depth
    mix = separator.prepare_mix(input_file)
    
    # Verify bit depth was detected correctly
    print(f"Detected bit depth: {separator.input_bit_depth}")
    print(f"Detected subtype: {separator.input_subtype}")
    
    assert separator.input_bit_depth == 24, f"Expected 24-bit, got {separator.input_bit_depth}"
    assert 'PCM_24' in separator.input_subtype, f"Expected PCM_24, got {separator.input_subtype}"
    
    print("✅ Test passed: 24-bit audio correctly detected")


def test_32bit_detection(temp_audio_dir, mock_separator_config):
    """Test that 32-bit audio files are correctly detected."""
    print("\n>>> TEST: 32-bit detection")
    
    # Create a 32-bit test audio file
    input_file = os.path.join(temp_audio_dir, "test_32bit.wav")
    create_test_audio_file(input_file, bit_depth=32)
    
    # Create CommonSeparator instance
    separator = CommonSeparator(mock_separator_config)
    
    # Call prepare_mix to detect bit depth
    mix = separator.prepare_mix(input_file)
    
    # Verify bit depth was detected correctly
    print(f"Detected bit depth: {separator.input_bit_depth}")
    print(f"Detected subtype: {separator.input_subtype}")
    
    assert separator.input_bit_depth == 32, f"Expected 32-bit, got {separator.input_bit_depth}"
    assert 'PCM_32' in separator.input_subtype or 'FLOAT' in separator.input_subtype, f"Expected PCM_32 or FLOAT, got {separator.input_subtype}"
    
    print("✅ Test passed: 32-bit audio correctly detected")


def test_numpy_array_input_defaults_to_16bit(mock_separator_config):
    """Test that numpy array input defaults to 16-bit."""
    print("\n>>> TEST: Numpy array input defaults to 16-bit")
    
    # Create a mock numpy array (stereo audio)
    mock_audio = np.random.rand(1000, 2).astype(np.float32)
    
    # Create CommonSeparator instance
    separator = CommonSeparator(mock_separator_config)
    
    # Call prepare_mix with numpy array
    mix = separator.prepare_mix(mock_audio)
    
    # Verify bit depth defaults to 16-bit
    print(f"Bit depth for numpy input: {separator.input_bit_depth}")
    assert separator.input_bit_depth == 16, f"Expected 16-bit default, got {separator.input_bit_depth}"
    
    print("✅ Test passed: Numpy array input defaults to 16-bit")


def test_bit_depth_preserved_across_multiple_files(temp_audio_dir, mock_separator_config):
    """Test that bit depth is correctly updated when processing multiple files."""
    print("\n>>> TEST: Bit depth updated across multiple files")
    
    # Create test files with different bit depths
    input_16bit = os.path.join(temp_audio_dir, "test_16bit.wav")
    input_24bit = os.path.join(temp_audio_dir, "test_24bit.wav")
    
    create_test_audio_file(input_16bit, bit_depth=16)
    create_test_audio_file(input_24bit, bit_depth=24)
    
    # Create CommonSeparator instance
    separator = CommonSeparator(mock_separator_config)
    
    # Process 16-bit file
    mix1 = separator.prepare_mix(input_16bit)
    assert separator.input_bit_depth == 16
    print(f"After 16-bit file: bit depth = {separator.input_bit_depth}")
    
    # Process 24-bit file
    mix2 = separator.prepare_mix(input_24bit)
    assert separator.input_bit_depth == 24
    print(f"After 24-bit file: bit depth = {separator.input_bit_depth}")
    
    # Process 16-bit file again
    mix3 = separator.prepare_mix(input_16bit)
    assert separator.input_bit_depth == 16
    print(f"After 16-bit file again: bit depth = {separator.input_bit_depth}")
    
    print("✅ Test passed: Bit depth correctly updated across multiple files")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

