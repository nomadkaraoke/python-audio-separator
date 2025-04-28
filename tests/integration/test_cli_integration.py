import os
import subprocess
import pytest
from pathlib import Path
from tests.utils import generate_reference_images, compare_images


@pytest.fixture(name="input_file")
def fixture_input_file():
    """Fixture providing the test input audio file path."""
    return "tests/inputs/mardy20s.flac"


@pytest.fixture(name="reference_dir")
def fixture_reference_dir():
    """Fixture providing the reference images directory path."""
    return "tests/inputs/reference"


@pytest.fixture(name="cleanup_output_files")
def fixture_cleanup_output_files():
    """Fixture to clean up output files before and after test."""
    # This list will be populated by the test functions
    output_files = []

    # Yield to allow the test to run and add files to the list
    yield output_files

    # Clean up output files after test
    for file in output_files:
        if os.path.exists(file):
            print(f"Test output file exists: {file}")
            os.remove(file)


def run_separation_test(model, audio_path, expected_files):
    """Helper function to run a separation test with a specific model."""
    # Clean up any existing output files before the test
    for file in expected_files:
        if os.path.exists(file):
            print(f"Deleting existing test output file {file}")
            os.remove(file)

    # Run the CLI command
    result = subprocess.run(["audio-separator", "-m", model, audio_path], capture_output=True, text=True, check=False)  # Explicitly set check to False as we handle errors manually

    # Check that the command completed successfully
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"

    # Check that the output files were created
    for file in expected_files:
        assert os.path.exists(file), f"Output file {file} was not created"
        assert os.path.getsize(file) > 0, f"Output file {file} is empty"

    return result


def validate_audio_output(output_file, reference_dir, waveform_threshold=0.999, spectrogram_threshold=None):
    """Validate an audio output file by comparing its waveform and spectrogram with reference images.

    Args:
        output_file: Path to the audio output file
        reference_dir: Directory containing reference images
        waveform_threshold: Minimum similarity required for waveform images (0.0-1.0)
        spectrogram_threshold: Minimum similarity for spectrogram images (0.0-1.0), defaults to waveform_threshold if None

    Returns:
        Tuple of booleans: (waveform_match, spectrogram_match)
    """
    # If spectrogram threshold not specified, use the same as waveform threshold
    if spectrogram_threshold is None:
        spectrogram_threshold = waveform_threshold

    # Create temporary directory for generated images
    temp_dir = os.path.join(os.path.dirname(output_file), "temp_images")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate waveform and spectrogram images for the output file
    output_filename = os.path.basename(output_file)
    name_without_ext = os.path.splitext(output_filename)[0]

    # Generate actual images
    actual_waveform_path, actual_spectrogram_path = generate_reference_images(output_file, temp_dir, prefix="actual_")

    # Path to expected reference images
    expected_waveform_path = os.path.join(reference_dir, f"expected_{name_without_ext}_waveform.png")
    expected_spectrogram_path = os.path.join(reference_dir, f"expected_{name_without_ext}_spectrogram.png")

    # Check if reference images exist
    if not os.path.exists(expected_waveform_path) or not os.path.exists(expected_spectrogram_path):
        print(f"Warning: Reference images not found for {output_file}")
        print(f"Expected: {expected_waveform_path} and {expected_spectrogram_path}")
        return False, False

    # Compare waveform images
    waveform_similarity, waveform_match = compare_images(expected_waveform_path, actual_waveform_path, min_similarity_threshold=waveform_threshold)

    # Compare spectrogram images
    spectrogram_similarity, spectrogram_match = compare_images(expected_spectrogram_path, actual_spectrogram_path, min_similarity_threshold=spectrogram_threshold)

    print(f"Validation results for {output_file}:\n")
    print(f"  Waveform similarity: {waveform_similarity:.4f} (match: {waveform_match}, threshold: {waveform_threshold:.2f})\n")
    print(f"  Spectrogram similarity: {spectrogram_similarity:.4f} (match: {spectrogram_match}, threshold: {spectrogram_threshold:.2f})\n")

    # Cleanup temp images (optional, uncomment if needed)
    # os.remove(actual_waveform_path)
    # os.remove(actual_spectrogram_path)

    return waveform_match, spectrogram_match


# Default similarity threshold to use for most models
DEFAULT_SIMILARITY_THRESHOLDS = (0.90, 0.80)  # (waveform_threshold, spectrogram_threshold)

# Model-specific similarity thresholds
# Use lower thresholds for models that show more variation between runs
MODEL_SIMILARITY_THRESHOLDS = {
    # Format: (waveform_threshold, spectrogram_threshold)
    "htdemucs_6s.yaml": (0.90, 0.70)  # Demucs multi-stem output (e.g. "Other" and "Piano") is a lot more variable
}


# Parameterized test for multiple models
MODEL_PARAMS = [
    # (model_filename, expected_output_filenames)
    ("kuielab_b_vocals.onnx", ["mardy20s_(Instrumental)_kuielab_b_vocals.flac", "mardy20s_(Vocals)_kuielab_b_vocals.flac"]),
    ("MGM_MAIN_v4.pth", ["mardy20s_(Instrumental)_MGM_MAIN_v4.flac", "mardy20s_(Vocals)_MGM_MAIN_v4.flac"]),
    ("UVR-MDX-NET-Inst_HQ_4.onnx", ["mardy20s_(Instrumental)_UVR-MDX-NET-Inst_HQ_4.flac", "mardy20s_(Vocals)_UVR-MDX-NET-Inst_HQ_4.flac"]),
    ("2_HP-UVR.pth", ["mardy20s_(Instrumental)_2_HP-UVR.flac", "mardy20s_(Vocals)_2_HP-UVR.flac"]),
    (
        "htdemucs_6s.yaml",
        [
            "mardy20s_(Vocals)_htdemucs_6s.flac",
            "mardy20s_(Drums)_htdemucs_6s.flac",
            "mardy20s_(Bass)_htdemucs_6s.flac",
            "mardy20s_(Other)_htdemucs_6s.flac",
            "mardy20s_(Guitar)_htdemucs_6s.flac",
            "mardy20s_(Piano)_htdemucs_6s.flac",
        ],
    ),
    ("model_bs_roformer_ep_937_sdr_10.5309.ckpt", ["mardy20s_(Drum-Bass)_model_bs_roformer_ep_937_sdr_10.flac", "mardy20s_(No Drum-Bass)_model_bs_roformer_ep_937_sdr_10.flac"]),
    ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", ["mardy20s_(Instrumental)_model_bs_roformer_ep_317_sdr_12.flac", "mardy20s_(Vocals)_model_bs_roformer_ep_317_sdr_12.flac"]),
]


@pytest.mark.parametrize("model,expected_files", MODEL_PARAMS)
def test_model_separation(model, expected_files, input_file, reference_dir, cleanup_output_files):
    """Parameterized test for multiple model files."""
    # Add files to the cleanup list
    cleanup_output_files.extend(expected_files)

    # Run the test
    run_separation_test(model, input_file, expected_files)

    # Validate the output audio files
    print(f"\nValidating output files for model {model}...")

    # Get model-specific similarity threshold or use default
    threshold = MODEL_SIMILARITY_THRESHOLDS.get(model, DEFAULT_SIMILARITY_THRESHOLDS)

    # Unpack thresholds - DEFAULT_SIMILARITY_THRESHOLDS is now always a tuple
    waveform_threshold, spectrogram_threshold = threshold

    print(f"Using thresholds - waveform: {waveform_threshold}, spectrogram: {spectrogram_threshold} for model {model}")

    for output_file in expected_files:
        # Skip validation if reference images are not required (set environment variable to skip)
        if os.environ.get("SKIP_AUDIO_VALIDATION") == "1":
            print(f"Skipping audio validation for {output_file} (SKIP_AUDIO_VALIDATION=1)")
            continue

        waveform_match, spectrogram_match = validate_audio_output(output_file, reference_dir, waveform_threshold=waveform_threshold, spectrogram_threshold=spectrogram_threshold)

        # Assert that the output matches the reference
        assert waveform_match, f"Waveform for {output_file} does not match the reference"
        assert spectrogram_match, f"Spectrogram for {output_file} does not match the reference"
