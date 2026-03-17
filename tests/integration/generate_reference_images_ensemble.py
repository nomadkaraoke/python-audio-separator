#!/usr/bin/env python3
"""
Script to generate reference waveform and spectrogram images for ensemble preset outputs.

Usage:
    1. Run all ensemble presets to generate output files:
       python -c "
       from audio_separator.separator import Separator
       presets = ['instrumental_clean', 'instrumental_full', 'instrumental_balanced',
                  'instrumental_low_resource', 'vocal_balanced', 'vocal_clean',
                  'vocal_full', 'vocal_rvc', 'karaoke']
       for p in presets:
           sep = Separator(output_dir='tests/inputs/ensemble_outputs', output_format='FLAC', ensemble_preset=p)
           sep.load_model()
           sep.separate('tests/inputs/mardy20s.flac')
       "

    2. Run this script to generate reference images from the outputs:
       python tests/integration/generate_reference_images_ensemble.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.utils import generate_reference_images
from tests.integration.test_ensemble_integration import ENSEMBLE_PRESET_PARAMS


def main():
    """Generate reference images for all expected ensemble preset outputs."""
    inputs_dir = Path(__file__).resolve().parent.parent / "inputs"
    reference_dir = inputs_dir / "reference"
    os.makedirs(reference_dir, exist_ok=True)

    # Check if output files exist — they need to be generated first
    # Look in a few common locations
    search_dirs = [
        "/tmp/ensemble-all-presets",
        str(inputs_dir / "ensemble_outputs"),
    ]

    output_dir = None
    for d in search_dirs:
        if os.path.isdir(d):
            output_dir = d
            break

    if output_dir is None:
        print("ERROR: No ensemble output files found.")
        print("First run the ensemble presets to generate output files.")
        print("Output files should be in one of:")
        for d in search_dirs:
            print(f"  {d}")
        sys.exit(1)

    print(f"Using output files from: {output_dir}")
    print(f"Generating reference images in: {reference_dir}\n")

    generated = 0
    missing = 0

    for preset, expected_files in ENSEMBLE_PRESET_PARAMS:
        print(f"Preset: {preset}")
        for output_filename in expected_files:
            file_path = os.path.join(output_dir, output_filename)

            if os.path.exists(file_path):
                print(f"  Generating images for: {output_filename}")
                generate_reference_images(file_path, str(reference_dir), prefix="expected_")
                generated += 2  # waveform + spectrogram
            else:
                print(f"  WARNING: Missing output file: {file_path}")
                missing += 1

    print(f"\nDone! Generated {generated} reference images.")
    if missing > 0:
        print(f"WARNING: {missing} output files were missing.")


if __name__ == "__main__":
    main()
