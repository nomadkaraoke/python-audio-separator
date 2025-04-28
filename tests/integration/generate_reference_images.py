#!/usr/bin/env python3
"""
Script to generate reference waveform and spectrogram images for model outputs.
This script should be run whenever the expected output files change.
"""

import os
import sys
import glob
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.utils import generate_reference_images
from tests.integration.test_cli_integration import MODEL_PARAMS

def main():
    """Generate reference images for all expected model outputs."""
    print("Generating reference images for expected model outputs...")
    
    # Get the input file path
    inputs_dir = Path(__file__).resolve().parent.parent / "inputs"
    input_file = inputs_dir / "mardy20s.flac"
    
    # Create reference directory if it doesn't exist
    reference_dir = inputs_dir / "reference"
    os.makedirs(reference_dir, exist_ok=True)
    
    # First, generate reference images for the input file
    print(f"Generating reference images for input file: {input_file}")
    generate_reference_images(str(input_file), str(reference_dir), prefix="expected_")
    
    # Then, generate reference images for each expected output file
    for model, expected_files in MODEL_PARAMS:
        for output_file in expected_files:
            file_path = os.path.join(str(inputs_dir), output_file)
            
            # Check if the file exists
            if os.path.exists(file_path):
                print(f"Generating reference images for output file: {output_file}")
                generate_reference_images(file_path, str(reference_dir), prefix="expected_")
            else:
                print(f"Warning: Output file does not exist: {file_path}")
                print(f"You may need to run the CLI command first to generate the output files.")
    
    print("Done generating reference images.")

if __name__ == "__main__":
    main() 