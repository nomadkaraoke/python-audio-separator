#!/usr/bin/env python3
"""
Script to generate reference waveform and spectrogram images for 24-bit model outputs.
This script should be run after creating the expected output files with 24-bit audio.

Usage:
    1. First run audio-separator on the 24-bit input file with each model:
       audio-separator -m model_bs_roformer_ep_317_sdr_12.9755.ckpt tests/inputs/fallen24bit20s.flac
       audio-separator -m mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt tests/inputs/fallen24bit20s.flac
       audio-separator -m MGM_MAIN_v4.pth tests/inputs/fallen24bit20s.flac
    
    2. Move the output files to tests/inputs/
    
    3. Run this script to generate reference images:
       python tests/integration/generate_reference_images_24bit.py
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.utils import generate_reference_images
from tests.integration.test_24bit_preservation import MODEL_PARAMS_24BIT


def main():
    """Generate reference images for all expected 24-bit model outputs."""
    print("="*60)
    print("Generating Reference Images for 24-bit Model Outputs")
    print("="*60)
    
    # Get the input file path
    inputs_dir = Path(__file__).resolve().parent.parent / "inputs"
    input_file = inputs_dir / "fallen24bit20s.flac"
    
    # Create reference directory if it doesn't exist
    reference_dir = inputs_dir / "reference"
    os.makedirs(reference_dir, exist_ok=True)
    
    # First, generate reference images for the 24-bit input file
    if input_file.exists():
        print(f"\n✓ Generating reference images for input file: {input_file}")
        generate_reference_images(str(input_file), str(reference_dir), prefix="expected_")
        print(f"  Created: expected_fallen24bit20s_waveform.png")
        print(f"  Created: expected_fallen24bit20s_spectrogram.png")
    else:
        print(f"\n✗ Error: Input file does not exist: {input_file}")
        print(f"  Please create the 24-bit test file first.")
        return 1
    
    # Then, generate reference images for each expected output file
    print(f"\nGenerating reference images for model outputs...")
    missing_files = []
    created_files = []
    
    for model, expected_files in MODEL_PARAMS_24BIT:
        print(f"\n{'-'*60}")
        print(f"Model: {model}")
        print(f"{'-'*60}")
        
        for output_file in expected_files:
            file_path = os.path.join(str(inputs_dir), output_file)
            
            # Check if the file exists
            if os.path.exists(file_path):
                print(f"✓ Processing: {output_file}")
                waveform_path, spectrogram_path = generate_reference_images(
                    file_path, str(reference_dir), prefix="expected_"
                )
                output_basename = os.path.splitext(output_file)[0]
                print(f"  Created: expected_{output_basename}_waveform.png")
                print(f"  Created: expected_{output_basename}_spectrogram.png")
                created_files.extend([waveform_path, spectrogram_path])
            else:
                print(f"✗ Missing: {output_file}")
                missing_files.append((model, output_file))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"✓ Created {len(created_files)} reference images")
    
    if missing_files:
        print(f"\n⚠  Warning: {len(missing_files)} output files were not found:")
        print(f"\nTo generate missing files, run these commands:")
        print()
        for model, _ in missing_files:
            # Only print each model once
            if missing_files.index((model, _)) == [m for m, _ in missing_files].index(model):
                print(f"  audio-separator -m {model} tests/inputs/fallen24bit20s.flac")
        print(f"\nThen move the output files to tests/inputs/ and run this script again.")
        print()
    else:
        print(f"\n✅ All reference images generated successfully!")
        print(f"\nYou can now run the 24-bit preservation tests:")
        print(f"  pytest tests/integration/test_24bit_preservation.py -v")
    
    print(f"{'='*60}\n")
    return 0 if not missing_files else 1


if __name__ == "__main__":
    sys.exit(main())

