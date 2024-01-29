import pytest
import logging
from audio_separator.utils.cli import main
from unittest import mock
from unittest.mock import patch, MagicMock


# Common fixture for expected arguments
@pytest.fixture
def common_expected_args():
    return {
        "log_formatter": mock.ANY,
        "log_level": logging.INFO,
        "model_file_dir": "/tmp/audio-separator-models/",
        "output_dir": None,
        "output_format": "FLAC",
        "denoise_enabled": False,
        "normalization_threshold": 0.9,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": 44100,
        "hop_length": 1024,
        "segment_size": 256,
        "overlap": 0.25,
        "batch_size": 1,
    }


# Test the CLI with no arguments
def test_cli_no_args(capsys):
    with patch("sys.argv", ["cli.py"]), pytest.raises(SystemExit):
        # Call the main function in cli.py
        main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out


# Test the CLI with a specific audio file
def test_cli_with_audio_file(capsys):
    test_args = ["cli.py", "test_audio.mp3"]
    with patch("audio_separator.separator.Separator.separate") as mock_separate:
        mock_separate.return_value = ["output_file.mp3"]
        with patch("sys.argv", test_args):
            # Call the main function in cli.py
            main()
    # Check if the separate
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
def test_cli_model_name_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--model_name=Custom_Model"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
            mock_separator_instance.load_model.assert_called_once_with("Custom_Model")


# Test using output directory argument
def test_cli_output_dir_argument(common_expected_args):
    test_args = ["cli.py", "test_audio.mp3", "--output_dir=/custom/output/dir"]
    with patch("sys.argv", test_args):
        with patch("audio_separator.separator.Separator") as mock_separator:
            mock_separator_instance = mock_separator.return_value
            mock_separator_instance.separate.return_value = ["output_file.mp3"]
            main()

            # Update expected args for this specific test
            common_expected_args["output_dir"] = "/custom/output/dir"

            # Assertions
            mock_separator.assert_called_once_with(**common_expected_args)
