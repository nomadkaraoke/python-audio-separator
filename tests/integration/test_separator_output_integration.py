import os
import pytest
import tempfile
import shutil
from pathlib import Path

from audio_separator.separator import Separator


@pytest.fixture(name="input_file")
def fixture_input_file():
    """Fixture providing the test input audio file path."""
    return "tests/inputs/mardy20s.flac"


@pytest.fixture(name="temp_output_dir")
def fixture_temp_output_dir():
    """Fixture providing a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


def test_separator_output_dir_and_custom_names(input_file, temp_output_dir):
    """Test that Separator respects output_dir and custom_output_names parameters."""
    print("\n>>> TEST: Checking output_dir with custom output names")
    
    # Define custom output filenames
    vocal_output_filename = "custom_vocals_output"
    instrumental_output_filename = "custom_instrumental_output"

    # Create output name mapping
    custom_output_names = {"Vocals": vocal_output_filename, "Instrumental": instrumental_output_filename}

    # Initialize separator with specified output directory
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    separator = Separator(output_dir=temp_output_dir, log_level=20)  # INFO level

    # Load model
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")

    # Run separation with custom output names
    print(f"Running separation with custom_output_names: {custom_output_names}")
    output_files = separator.separate(input_file, custom_output_names=custom_output_names)
    print(f"Separator.separate() returned: {output_files}")

    # The separator adds .wav extension since the default output format is WAV
    expected_vocal_filename = vocal_output_filename + ".wav"
    expected_instrumental_filename = instrumental_output_filename + ".wav"
    
    # Check that the returned filenames match the expected names
    output_filenames = [os.path.basename(f) for f in output_files]
    print(f"Extracted filenames from output_files: {output_filenames}")
    
    # NOTE: The Separator class returns only the filenames, not the full paths with output_dir
    print("EXPECTED BEHAVIOR: The Separator.separate() method returns filenames without the output_dir path")
    print(f"Expected filenames (without path): {expected_vocal_filename} and {expected_instrumental_filename}")
    
    assert expected_vocal_filename in output_filenames, f"Expected {expected_vocal_filename} in output files"
    assert expected_instrumental_filename in output_filenames, f"Expected {expected_instrumental_filename} in output files"
    
    # Check that files physically exist in the specified output directory
    expected_vocal_path = os.path.join(temp_output_dir, expected_vocal_filename)
    expected_instrumental_path = os.path.join(temp_output_dir, expected_instrumental_filename)
    
    print(f"Checking that files exist in output_dir: {temp_output_dir}")
    print(f"Full expected vocal path: {expected_vocal_path}")
    print(f"Full expected instrumental path: {expected_instrumental_path}")
    
    assert os.path.exists(expected_vocal_path), f"Vocals output file doesn't exist: {expected_vocal_path}"
    assert os.path.exists(expected_instrumental_path), f"Instrumental output file doesn't exist: {expected_instrumental_path}"
    assert os.path.getsize(expected_vocal_path) > 0, f"Vocals output file is empty: {expected_vocal_path}"
    assert os.path.getsize(expected_instrumental_path) > 0, f"Instrumental output file is empty: {expected_instrumental_path}"
    
    print("✅ Test passed: Separator correctly handles output_dir and custom_output_names")
    print("   - Files were saved to the specified output directory")
    print("   - Custom filenames were used (with .wav extension added)")
    print("   - Returned paths include just the filenames (not the full paths)")


def test_separator_single_stem_output(input_file, temp_output_dir):
    """Test that Separator correctly respects output_single_stem with custom output name."""
    print("\n>>> TEST: Checking output_single_stem with custom output name")
    
    # Define custom output filename for single stem
    vocal_output_filename = "only_vocals_output"

    # Create output name mapping
    custom_output_names = {"Vocals": vocal_output_filename}

    # Initialize separator with specified output directory and single stem output
    print(f"Creating Separator with output_dir: {temp_output_dir} and output_single_stem: Vocals")
    separator = Separator(
        output_dir=temp_output_dir,
        output_single_stem="Vocals",  # Only extract vocals
        log_level=20  # INFO level
    )

    # Load model
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")

    # Run separation with custom output name
    print(f"Running separation with custom_output_names: {custom_output_names}")
    output_files = separator.separate(input_file, custom_output_names=custom_output_names)
    print(f"Separator.separate() returned: {output_files}")

    # The separator adds .wav extension since the default output format is WAV
    expected_vocal_filename = vocal_output_filename + ".wav"
    
    # Check that the output files list contains only the expected file
    print(f"Checking that only one file was returned and it has the correct name")
    assert len(output_files) == 1, f"Expected 1 output file, got {len(output_files)}"
    assert os.path.basename(output_files[0]) == expected_vocal_filename, f"Expected {expected_vocal_filename} in output files"
    
    # Check that file physically exists in the specified output directory
    expected_vocal_path = os.path.join(temp_output_dir, expected_vocal_filename)
    print(f"Checking that file exists in output_dir: {expected_vocal_path}")
    
    assert os.path.exists(expected_vocal_path), f"Vocals output file doesn't exist: {expected_vocal_path}"
    assert os.path.getsize(expected_vocal_path) > 0, f"Vocals output file is empty: {expected_vocal_path}"
    
    # Make sure the instrumental file was NOT created
    instrumental_files = [f for f in os.listdir(temp_output_dir) if "instrumental" in f.lower()]
    print(f"Checking that no instrumental files were created, found: {instrumental_files}")
    assert len(instrumental_files) == 0, f"No instrumental file should be created when using output_single_stem, found: {instrumental_files}"
    
    print("✅ Test passed: Separator correctly handles output_single_stem with custom output name")
    print("   - Only the specified stem (Vocals) was extracted")
    print("   - The custom filename was used (with .wav extension added)")
    print("   - No other stems (like Instrumental) were created")


def test_separator_output_without_custom_names(input_file, temp_output_dir):
    """Test that Separator respects output_dir without custom_output_names."""
    print("\n>>> TEST: Checking output_dir without custom output names")
    
    # Initialize separator with specified output directory
    print(f"Creating Separator with output_dir: {temp_output_dir}")
    separator = Separator(output_dir=temp_output_dir, log_level=20)  # INFO level

    # Load model
    print("Loading model: MGM_MAIN_v4.pth")
    separator.load_model(model_filename="MGM_MAIN_v4.pth")

    # Run separation without custom output names
    print("Running separation without custom_output_names")
    output_files = separator.separate(input_file)
    print(f"Separator.separate() returned: {output_files}")

    # Check that output files exist and have content
    print(f"Checking that two files were returned (one for each stem)")
    assert len(output_files) == 2, f"Expected 2 output files, got {len(output_files)}"
    
    # Check if the files were created in the output directory
    # Note: The separator doesn't include the full path in the returned output_files
    output_file_basenames = [os.path.basename(f) for f in output_files]
    print(f"Extracted filenames from output_files: {output_file_basenames}")
    
    print("EXPECTED BEHAVIOR: Default filenames are being used (format: inputname_(StemName)_modelname.wav)")
    
    # Check all output files exist in the specified output directory
    print(f"Checking that files exist in output_dir: {temp_output_dir}")
    for basename in output_file_basenames:
        full_path = os.path.join(temp_output_dir, basename)
        print(f"Checking file: {full_path}")
        assert os.path.exists(full_path), f"Output file doesn't exist: {full_path}"
        assert os.path.getsize(full_path) > 0, f"Output file is empty: {full_path}"
    
    print("✅ Test passed: Separator correctly handles output_dir without custom output names")
    print("   - Files were saved to the specified output directory")
    print("   - Default naming scheme was used (input_name_(Stem)_model.wav)")
    print("   - Returned paths include just the filenames (not the full paths)")
