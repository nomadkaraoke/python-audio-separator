import pytest
import logging
from audio_separator.utils.cli import main
import subprocess
from unittest import mock
from unittest.mock import patch, MagicMock, mock_open


# Common fixture for expected arguments
@pytest.fixture
def common_expected_args():
    return {
        "log_formatter": mock.ANY,
        "log_level": logging.INFO,
        "model_file_dir": "/tmp/audio-separator-models/",
        "output_dir": None,
        "output_format": "FLAC",
        "output_bitrate": None,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.6,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": 44100,
        "use_autocast": False,
        "use_soundfile": False,
        "mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False},
        "vr_params": {"batch_size": 1, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
        "demucs_params": {"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True},
        "mdxc_params": {"segment_size": 256, "batch_size": 1, "overlap": 8, "override_model_segment_size": False, "pitch_shift": 0},
    }


# Test the CLI with version argument using subprocess
def test_cli_version_subprocess():
    # Run the CLI script with the '--version' argument
    result = subprocess.run(["poetry", "run", "audio-separator", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "audio-separator" in result.stdout

    # Test with the short version flag '-v'
    result = subprocess.run(["poetry", "run", "audio-separator", "-v"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "audio-separator" in result.stdout


# Test the CLI with no arguments
def test_cli_no_args(capsys):
    result = subprocess.run(["poetry", "run", "audio-separator"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "usage:" in result.stdout


# Test with multiple filename arguments
def test_cli_multiple_filenames():
    test_args = ["cli.py", "test1.mp3", "test2.mp3"]

    # Mock the open function to prevent actual file operations
    mock_file = mock_open()

    # Create a mock logger
    mock_logger = MagicMock()

    # Patch multiple functions to prevent actual file operations and separations
    with patch("sys.argv", test_args), patch("builtins.open", mock_file), patch("audio_separator.separator.Separator.separate") as mock_separate, patch(
        "audio_separator.separator.Separator.load_model"
    ), patch("logging.getLogger", return_value=mock_logger):

        # Mock the separate method to return some dummy output
        mock_separate.return_value = ["output_file1.mp3", "output_file2.mp3"]

        # Call the main function
        main()

        # Check if separate was called twice (once for each input file)
        assert mock_separate.call_count == 2

        # Check if the logger captured information about both files
        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("test1.mp3" in msg and "test2.mp3" in msg for msg in log_messages)
        assert any("Separation complete" in msg for msg in log_messages)


# Test the CLI with a specific audio file
def test_cli_with_audio_file(capsys, common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--model_filename=UVR-MDX-NET-Inst_HQ_4.onnx"]
    with patch("audio_separator.separator.Separator.separate") as mock_separate:
        mock_separate.return_value = ["output_file.mp3"]
        with patch("sys.argv", test_args):
            # Call the main function in cli.py
            main()

    # Update expected args for this specific test
    common_expected_args["model_file_dir"] = "/tmp/audio-separator-models/"

    # Check if the separate method was called with the correct arguments
    mock_separate.assert_called_once()

    # Assertions
    assert mock_separate.called


# Test the CLI with invalid log level
def test_cli_invalid_log_level():
    test_args = ["cli.py", "test_audio.mp3", "--log_level=invalid"]
    with patch("sys.argv", test_args):
        # Assert an attribute error is raised due to the invalid LogLevel
        with pytest.raises(AttributeError):
            # Call the main function in cli.py
            main()


# Test using model name argument
def test_cli_model_filename_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--model_filename=Custom_Model.onnx"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.load_model.assert_called_once_with(model_filename="Custom_Model.onnx")


# Test using output directory argument
def test_cli_output_dir_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--output_dir=/custom/output/dir"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["output_dir"] = "/custom/output/dir"

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using output format argument
def test_cli_output_format_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--output_format=MP3"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["output_format"] = "MP3"

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using normalization_threshold argument
def test_cli_normalization_threshold_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--normalization=0.75"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["normalization_threshold"] = 0.75

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using normalization_threshold argument
def test_cli_amplification_threshold_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--amplification=0.75"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["amplification_threshold"] = 0.75

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using single stem argument
def test_cli_single_stem_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--single_stem=instrumental"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["output_single_stem"] = "instrumental"

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using invert spectrogram argument
def test_cli_invert_spectrogram_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--invert_spect"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["invert_using_spec"] = True

            # Assertions
            mock_separator.assert_called_once_with(**expected_args)


# Test using use_autocast argument
def test_cli_use_autocast_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_autocast"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            expected_args = common_expected_args.copy()
            expected_args["use_autocast"] = True

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)


# Test using use_autocast argument
def test_cli_use_autocast_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--use_autocast"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            common_expected_args["use_autocast"] = True

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)


# Test using primary_output_name argument
def test_cli_primary_output_name_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--primary_output_name=custom_primary_output"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.separate.assert_called_once_with("test_audio.mp3", primary_output_name="custom_primary_output", secondary_output_name=None)


# Test using secondary_output_name argument
def test_cli_secondary_output_name_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--secondary_output_name=custom_secondary_output"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.separate.assert_called_once_with("test_audio.mp3", primary_output_name=None, secondary_output_name="custom_secondary_output")


# Test using both primary_output_name and secondary_output_name arguments
def test_cli_both_output_names_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--primary_output_name=custom_primary_output", "--secondary_output_name=custom_secondary_output"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.separate.assert_called_once_with("test_audio.mp3", primary_output_name="custom_primary_output", secondary_output_name="custom_secondary_output")
