import pytest
from audio_separator.utils.cli import main
from unittest.mock import patch, MagicMock

# Test the CLI with no arguments
def test_cli_no_args(capsys):
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out

# Test the CLI with a specific audio file
def test_cli_with_audio_file(capsys):
    test_args = ["cli.py", "test_audio.mp3"]
    with patch("audio_separator.separator.Separator.separate") as mock_separate:
        mock_separate.return_value = ["output_file.mp3"]
        with patch("sys.argv", test_args):
            main()
    # Check if the function runs without errors, as capturing output might not be reliable
    assert mock_separate.called

# Test the CLI with invalid log level
def test_cli_invalid_log_level():
    test_args = ["cli.py", "test_audio.mp3", "--log_level=invalid"]
    with patch("sys.argv", test_args):
        with pytest.raises(AttributeError):
            main()
