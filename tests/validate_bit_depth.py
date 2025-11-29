#!/usr/bin/env python3
"""
Test script to validate that 24-bit audio is correctly preserved.

This script creates a 24-bit test file, processes it through the separator,
and verifies both the bit depth and audio content are correct.
"""

import os
import sys
import tempfile
import shutil
import soundfile as sf
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_separator.separator.common_separator import CommonSeparator
from unittest.mock import Mock


def create_test_audio_file(output_path, sample_rate=44100, duration=2.0, bit_depth=24):
    """Create a test audio file with a specific bit depth."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    stereo_audio = np.column_stack([audio, audio])
    
    subtype = 'PCM_24' if bit_depth == 24 else 'PCM_16'
    sf.write(output_path, stereo_audio, sample_rate, subtype=subtype)
    return output_path


def get_audio_info(file_path):
    """Get audio file information."""
    info = sf.info(file_path)
    audio, sr = sf.read(file_path)
    return {
        "duration": info.duration,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "frames": info.frames,
        "audio": audio,
    }


def main():
    print("="*60)
    print("Testing 24-bit Audio Preservation")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create 24-bit input file
        input_file = os.path.join(temp_dir, "input_24bit.flac")
        create_test_audio_file(input_file, bit_depth=24, duration=1.0)
        
        print(f"\n1. Created input file: {input_file}")
        input_info = get_audio_info(input_file)
        print(f"   Duration: {input_info['duration']:.2f}s")
        print(f"   Sample rate: {input_info['samplerate']} Hz")
        print(f"   Bit depth: {input_info['subtype']}")
        print(f"   Frames: {input_info['frames']}")
        print(f"   Audio shape: {input_info['audio'].shape}")
        print(f"   Audio range: [{input_info['audio'].min():.6f}, {input_info['audio'].max():.6f}]")
        
        # Create separator config
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
            "output_format": "flac",
            "output_bitrate": None,
            "normalization_threshold": 0.9,
            "amplification_threshold": 0.0,
            "enable_denoise": False,
            "output_single_stem": None,
            "invert_using_spec": False,
            "sample_rate": 44100,
            "use_soundfile": False,
        }
        
        # Create separator and prepare mix
        print("\n2. Processing through CommonSeparator...")
        separator = CommonSeparator(config)
        separator.audio_file_path = input_file
        mix = separator.prepare_mix(input_file)
        
        print(f"   Detected bit depth: {separator.input_bit_depth}-bit")
        print(f"   Mix shape: {mix.shape}")
        
        # Create output audio (simulated separation result - just pass through)
        output_audio = mix.T
        print(f"   Output audio shape: {output_audio.shape}")
        print(f"   Output audio range: [{output_audio.min():.6f}, {output_audio.max():.6f}]")
        
        # Write output file
        output_file = "output_24bit.flac"
        print(f"\n3. Writing output file...")
        separator.write_audio_pydub(output_file, output_audio)
        
        # Check output file
        full_output_path = os.path.join(temp_dir, output_file)
        if os.path.exists(full_output_path):
            output_info = get_audio_info(full_output_path)
            print(f"   Duration: {output_info['duration']:.2f}s")
            print(f"   Sample rate: {output_info['samplerate']} Hz")
            print(f"   Bit depth: {output_info['subtype']}")
            print(f"   Frames: {output_info['frames']}")
            print(f"   Audio shape: {output_info['audio'].shape}")
            print(f"   Audio range: [{output_info['audio'].min():.6f}, {output_info['audio'].max():.6f}]")
            
            # Validation
            print("\n4. Validation:")
            
            # Check bit depth
            if 'PCM_24' in output_info['subtype']:
                print("   ✅ Bit depth is 24-bit")
            else:
                print(f"   ❌ Bit depth is {output_info['subtype']}, expected PCM_24")
            
            # Check duration (allow small tolerance)
            duration_diff = abs(output_info['duration'] - input_info['duration'])
            if duration_diff < 0.01:  # Within 10ms
                print(f"   ✅ Duration matches (diff: {duration_diff*1000:.2f}ms)")
            else:
                print(f"   ❌ Duration mismatch: input={input_info['duration']:.2f}s, output={output_info['duration']:.2f}s (diff: {duration_diff:.2f}s)")
            
            # Check frame count
            if output_info['frames'] == input_info['frames']:
                print(f"   ✅ Frame count matches ({output_info['frames']} frames)")
            else:
                print(f"   ❌ Frame count mismatch: input={input_info['frames']}, output={output_info['frames']}")
            
            # Check audio is not just noise (should have recognizable signal)
            audio_std = np.std(output_info['audio'])
            if audio_std > 0.01:  # Has reasonable signal
                print(f"   ✅ Audio has signal (std: {audio_std:.6f})")
            else:
                print(f"   ❌ Audio appears to be silent or noise (std: {audio_std:.6f})")
            
            # Check audio correlation with input (should be similar since we're just passing through)
            # Truncate to shorter length for comparison
            min_len = min(len(input_info['audio']), len(output_info['audio']))
            correlation = np.corrcoef(input_info['audio'][:min_len, 0], output_info['audio'][:min_len, 0])[0, 1]
            if correlation > 0.99:
                print(f"   ✅ Audio content preserved (correlation: {correlation:.6f})")
            else:
                print(f"   ⚠️  Audio content may be corrupted (correlation: {correlation:.6f})")
            
            print("\n" + "="*60)
            if ('PCM_24' in output_info['subtype'] and 
                duration_diff < 0.01 and 
                output_info['frames'] == input_info['frames'] and
                audio_std > 0.01 and
                correlation > 0.99):
                print("✅ ALL TESTS PASSED!")
                print("="*60)
                return 0
            else:
                print("❌ SOME TESTS FAILED!")
                print("="*60)
                return 1
        else:
            print(f"   ❌ Output file not created: {full_output_path}")
            return 1
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    sys.exit(main())

