"""
Reproduce the ensemble + custom_output_names bug against the live API.

This script simulates exactly what karaoke-gen's audio_processor does:
1. Call the API with preset=instrumental_clean and custom_output_names
2. Download the results
3. Check if the expected filenames exist

Expected behavior (fixed): files named job123_mixed_vocals.flac and job123_mixed_instrumental.flac
Bug behavior (current prod): files named with original filename + _(Unknown)_ or _(Other)_

Usage:
    python tests/reproduce_ensemble_bug.py [--api-url URL]
"""
import json
import os
import sys
import tempfile

# Add the repo to path so we can import the API client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_separator.remote.api_client import AudioSeparatorAPIClient


def main():
    api_url = os.environ.get("AUDIO_SEPARATOR_API_URL")
    if not api_url:
        print("ERROR: Set AUDIO_SEPARATOR_API_URL environment variable")
        sys.exit(1)

    test_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "under_pressure_harmonies.flac")
    if not os.path.exists(test_audio):
        print(f"ERROR: Test audio file not found: {test_audio}")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="ensemble_bug_test_") as output_dir:
        print(f"API URL: {api_url}")
        print(f"Output dir: {output_dir}")
        print()

        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logger = logging.getLogger("test")

        client = AudioSeparatorAPIClient(api_url, logger)

        # This is exactly what karaoke-gen does in _process_audio_separation_remote
        file_prefix = "job123"  # Simulates job_id-based prefix
        custom_output_names = {
            "Vocals": f"{file_prefix}_mixed_vocals",
            "Instrumental": f"{file_prefix}_mixed_instrumental",
        }

        print("=" * 60)
        print("TEST: Preset + custom_output_names (reproduces karaoke-gen bug)")
        print(f"  preset: instrumental_clean")
        print(f"  custom_output_names: {custom_output_names}")
        print("=" * 60)
        print()

        result = client.separate_audio_and_wait(
            test_audio,
            preset="instrumental_clean",
            timeout=600,
            poll_interval=10,
            download=True,
            output_dir=output_dir,
            output_format="flac",
            custom_output_names=custom_output_names,
        )

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Status: {result.get('status')}")
        print(f"Downloaded files: {result.get('downloaded_files', [])}")
        print()

        # List what's actually in the output dir
        actual_files = os.listdir(output_dir)
        print(f"Files in output dir: {actual_files}")
        print()

        # Check for expected files
        fmt = "flac"
        expected_vocals = f"{file_prefix}_mixed_vocals.{fmt}"
        expected_instrumental = f"{file_prefix}_mixed_instrumental.{fmt}"

        vocals_exists = os.path.exists(os.path.join(output_dir, expected_vocals))
        instrumental_exists = os.path.exists(os.path.join(output_dir, expected_instrumental))

        print("EXPECTED FILE CHECK:")
        print(f"  {expected_vocals}: {'FOUND' if vocals_exists else 'MISSING'}")
        print(f"  {expected_instrumental}: {'FOUND' if instrumental_exists else 'MISSING'}")
        print()

        if vocals_exists and instrumental_exists:
            print("RESULT: PASS - custom_output_names working correctly")
            return 0
        else:
            print("RESULT: FAIL - custom_output_names NOT applied (bug reproduced)")
            print()
            print("Actual files downloaded:")
            for f in actual_files:
                size = os.path.getsize(os.path.join(output_dir, f))
                print(f"  {f} ({size / 1024:.1f} KB)")
            return 1


if __name__ == "__main__":
    sys.exit(main())
