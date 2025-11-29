#!/usr/bin/env python3
"""
Manual test script for bit depth preservation.

This script creates test audio files with different bit depths and verifies
that the separator preserves the bit depth in the output files.

Usage:
    python manual_test_bit_depth.py
"""

import os
import sys
import tempfile
import shutil
import soundfile as sf
import numpy as np

# Add parent directory to path to import audio_separator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_separator.separator.common_separator import CommonSeparator


def create_test_audio_file(output_path, sample_rate=44100, duration=2.0, bit_depth=16):
    """Create a test audio file with a specific bit depth."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    stereo_audio = np.column_stack([audio, audio])
    
    if bit_depth == 16:
        subtype = 'PCM_16'
    elif bit_depth == 24:
        subtype = 'PCM_24'
    elif bit_depth == 32:
        subtype = 'PCM_32'
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    sf.write(output_path, stereo_audio, sample_rate, subtype=subtype)
    print(f"✓ Created {bit_depth}-bit test file: {output_path}")
    return output_path


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


def test_bit_depth_preservation(bit_depth, use_soundfile=False):
    """Test bit depth preservation for a specific bit depth."""
    print(f"\n{'='*60}")
    print(f"Testing {bit_depth}-bit preservation {'with soundfile' if use_soundfile else 'with pydub'}")
    print(f"{'='*60}")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test input file
        input_file = os.path.join(temp_dir, f"input_{bit_depth}bit.wav")
        create_test_audio_file(input_file, bit_depth=bit_depth, duration=1.0)
        
        # Verify input bit depth
        input_bit = get_audio_bit_depth(input_file)
        print(f"✓ Input file verified as {input_bit}-bit")
        
        # Create CommonSeparator config
        from unittest.mock import Mock
        config = {
            "logger": Mock(),
            "log_level": 20,
            "torch_device": Mock(),
            "torch_device_cpu": Mock(),
            "torch_device_mps": Mock(),
            "onnx_execution_provider": Mock(),
            "model_name": "test_model",
            "model_path": "/path/to/model",
            "model_data": {"training": {"instruments": ["vocals", "other"]}},
            "output_dir": temp_dir,
            "output_format": "wav",
            "output_bitrate": None,
            "normalization_threshold": 0.9,
            "amplification_threshold": 0.0,
            "enable_denoise": False,
            "output_single_stem": None,
            "invert_using_spec": False,
            "sample_rate": 44100,
            "use_soundfile": use_soundfile,
        }
        
        # Create separator and prepare mix
        separator = CommonSeparator(config)
        separator.audio_file_path = input_file
        mix = separator.prepare_mix(input_file)
        
        print(f"✓ Detected input bit depth: {separator.input_bit_depth}-bit")
        print(f"✓ Detected input subtype: {separator.input_subtype}")
        
        # Create output audio (simulated separation result)
        output_audio = mix.T
        
        # Write output file
        output_file = f"output_{bit_depth}bit{'_sf' if use_soundfile else ''}.wav"
        if use_soundfile:
            separator.write_audio_soundfile(output_file, output_audio)
        else:
            separator.write_audio_pydub(output_file, output_audio)
        
        # Verify output bit depth
        full_output_path = os.path.join(temp_dir, output_file)
        if os.path.exists(full_output_path):
            output_bit = get_audio_bit_depth(full_output_path)
            print(f"✓ Output file verified as {output_bit}-bit")
            
            if output_bit == bit_depth:
                print(f"✅ SUCCESS: {bit_depth}-bit preservation works correctly!")
                return True
            else:
                print(f"❌ FAILURE: Expected {bit_depth}-bit, got {output_bit}-bit")
                return False
        else:
            print(f"❌ FAILURE: Output file not created")
            return False
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def main():
    """Run all bit depth preservation tests."""
    print("="*60)
    print("Manual Bit Depth Preservation Tests")
    print("="*60)
    
    results = []
    
    # Test with pydub backend
    results.append(("16-bit (pydub)", test_bit_depth_preservation(16, use_soundfile=False)))
    results.append(("24-bit (pydub)", test_bit_depth_preservation(24, use_soundfile=False)))
    results.append(("32-bit (pydub)", test_bit_depth_preservation(32, use_soundfile=False)))
    
    # Test with soundfile backend
    results.append(("16-bit (soundfile)", test_bit_depth_preservation(16, use_soundfile=True)))
    results.append(("24-bit (soundfile)", test_bit_depth_preservation(24, use_soundfile=True)))
    results.append(("32-bit (soundfile)", test_bit_depth_preservation(32, use_soundfile=True)))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    # Overall result
    all_passed = all(result for _, result in results)
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print(f"{'='*60}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

