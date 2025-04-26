import os
import subprocess
import pytest


@pytest.fixture(name="input_file")
def fixture_input_file():
    """Fixture providing the test input audio file path."""
    return "tests/inputs/mardy20s.flac"


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
            # os.remove(file)


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
def test_model_separation(model, expected_files, input_file, cleanup_output_files):
    """Parameterized test for multiple model files."""
    # Add files to the cleanup list
    cleanup_output_files.extend(expected_files)

    # Run the test
    run_separation_test(model, input_file, expected_files)
