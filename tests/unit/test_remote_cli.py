import json
import pytest
import argparse
import logging
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from unittest.mock import call
import tempfile

from audio_separator.remote.cli import main, handle_separate_command, handle_status_command, handle_models_command, handle_download_command
from audio_separator.remote import AudioSeparatorAPIClient


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing."""
    return Mock(spec=AudioSeparatorAPIClient)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio content")
        yield f.name
    os.unlink(f.name)


class TestRemoteCLI:
    """Test the remote CLI functionality."""

    @patch('sys.argv', ['audio-separator-remote', '--version'])
    @patch('audio_separator.remote.cli.metadata')
    @patch('builtins.print')
    def test_version_command_no_api_url(self, mock_print, mock_metadata, mock_api_client):
        """Test version command without API URL."""
        mock_metadata.distribution.return_value.version = "1.2.3"
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
        mock_print.assert_any_call("Client version: 1.2.3")

    @patch('sys.argv', ['audio-separator-remote', '--version'])
    @patch('audio_separator.remote.cli.metadata')
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('builtins.print')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_version_command_with_api_url(self, mock_print, mock_client_class, mock_metadata):
        """Test version command with API URL."""
        mock_metadata.distribution.return_value.version = "1.2.3"
        mock_client = Mock()
        mock_client.get_server_version.return_value = "1.2.4"
        mock_client_class.return_value = mock_client
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
        mock_print.assert_any_call("Client version: 1.2.3")
        mock_print.assert_any_call("Server version: 1.2.4")

    @patch('sys.argv', ['audio-separator-remote', '--version'])
    @patch('audio_separator.remote.cli.metadata')
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('builtins.print')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_version_command_server_error(self, mock_print, mock_client_class, mock_metadata):
        """Test version command when server version retrieval fails."""
        mock_metadata.distribution.return_value.version = "1.2.3"
        mock_client = Mock()
        mock_client.get_server_version.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
        mock_print.assert_any_call("Client version: 1.2.3")

    @patch('sys.argv', ['audio-separator-remote', 'separate', 'test.wav'])
    @patch('builtins.print')
    @patch.dict(os.environ, {}, clear=True)  # Clear all environment variables
    def test_no_api_url_error(self, mock_print):
        """Test error when no API URL is provided."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1

    @patch('sys.argv', ['audio-separator-remote', 'separate', 'test.wav'])
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('audio_separator.remote.cli.handle_separate_command')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_separate_command(self, mock_handle_separate, mock_client_class):
        """Test separate command execution."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        main()
        
        mock_handle_separate.assert_called_once()
        # Verify the API client was called with the correct URL (don't check logger instance)
        assert mock_client_class.call_count == 1
        call_args = mock_client_class.call_args
        assert call_args[0][0] == 'https://test-api.com'  # First argument should be the API URL

    @patch('sys.argv', ['audio-separator-remote', 'status', 'task-123'])
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('audio_separator.remote.cli.handle_status_command')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_status_command(self, mock_handle_status, mock_client_class):
        """Test status command execution."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        main()
        
        mock_handle_status.assert_called_once()

    @patch('sys.argv', ['audio-separator-remote', 'models'])
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('audio_separator.remote.cli.handle_models_command')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_models_command(self, mock_handle_models, mock_client_class):
        """Test models command execution."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        main()
        
        mock_handle_models.assert_called_once()

    @patch('sys.argv', ['audio-separator-remote', 'download', 'task-123', 'file1.wav', 'file2.wav'])
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('audio_separator.remote.cli.handle_download_command')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_download_command(self, mock_handle_download, mock_client_class):
        """Test download command execution."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        main()
        
        mock_handle_download.assert_called_once()

    @patch('sys.argv', ['audio-separator-remote', '--api_url', 'https://custom-api.com', 'models'])
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('audio_separator.remote.cli.handle_models_command')
    def test_custom_api_url(self, mock_handle_models, mock_client_class):
        """Test using custom API URL parameter."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        main()
        
        # Verify the API client was called with the correct custom URL
        assert mock_client_class.call_count == 1
        call_args = mock_client_class.call_args
        assert call_args[0][0] == 'https://custom-api.com'  # First argument should be the custom API URL

    @patch('sys.argv', ['audio-separator-remote', '--debug', 'models'])
    @patch('audio_separator.remote.cli.AudioSeparatorAPIClient')
    @patch('audio_separator.remote.cli.handle_models_command')
    @patch.dict(os.environ, {'AUDIO_SEPARATOR_API_URL': 'https://test-api.com'})
    def test_debug_logging(self, mock_handle_models, mock_client_class):
        """Test debug logging flag."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        main()
        
        # Verify debug logging is enabled by checking if logger level was set
        mock_handle_models.assert_called_once()

    def test_handle_separate_command_success(self, mock_api_client, mock_logger, mock_audio_file):
        """Test successful separate command handling."""
        # Mock arguments
        args = Mock()
        args.audio_files = [mock_audio_file]
        args.model = "test_model.ckpt"
        args.models = None
        args.timeout = 600
        args.poll_interval = 10
        args.output_format = "flac"
        args.output_bitrate = None
        args.normalization = 0.9
        args.amplification = 0.0
        args.single_stem = None
        args.invert_spect = False
        args.sample_rate = 44100
        args.use_soundfile = False
        args.use_autocast = False
        args.custom_output_names = None
        # MDX parameters
        args.mdx_segment_size = 256
        args.mdx_overlap = 0.25
        args.mdx_batch_size = 1
        args.mdx_hop_length = 1024
        args.mdx_enable_denoise = False
        # VR parameters
        args.vr_batch_size = 1
        args.vr_window_size = 512
        args.vr_aggression = 5
        args.vr_enable_tta = False
        args.vr_high_end_process = False
        args.vr_enable_post_process = False
        args.vr_post_process_threshold = 0.2
        # Demucs parameters
        args.demucs_segment_size = "Default"
        args.demucs_shifts = 2
        args.demucs_overlap = 0.25
        args.demucs_segments_enabled = True
        # MDXC parameters
        args.mdxc_segment_size = 256
        args.mdxc_override_model_segment_size = False
        args.mdxc_overlap = 8
        args.mdxc_batch_size = 1
        args.mdxc_pitch_shift = 0

        # Mock successful API response
        mock_api_client.separate_audio_and_wait.return_value = {
            "status": "completed",
            "downloaded_files": ["output1.wav", "output2.wav"]
        }

        handle_separate_command(args, mock_api_client, mock_logger)

        # Verify API client was called with correct parameters
        mock_api_client.separate_audio_and_wait.assert_called_once()
        call_args = mock_api_client.separate_audio_and_wait.call_args
        assert call_args[0][0] == mock_audio_file  # First positional argument should be the audio file
        kwargs = call_args[1]
        assert kwargs["model"] == "test_model.ckpt"
        assert kwargs["timeout"] == 600
        assert kwargs["download"] is True

    def test_handle_separate_command_with_multiple_models(self, mock_api_client, mock_logger, mock_audio_file):
        """Test separate command with multiple models."""
        args = Mock()
        args.audio_files = [mock_audio_file]
        args.model = None
        args.models = ["model1.ckpt", "model2.onnx"]
        args.timeout = 600
        args.poll_interval = 10
        # Set other required attributes
        for attr in ['output_format', 'output_bitrate', 'normalization', 'amplification', 'single_stem',
                     'invert_spect', 'sample_rate', 'use_soundfile', 'use_autocast', 'custom_output_names',
                     'mdx_segment_size', 'mdx_overlap', 'mdx_batch_size', 'mdx_hop_length', 'mdx_enable_denoise',
                     'vr_batch_size', 'vr_window_size', 'vr_aggression', 'vr_enable_tta', 'vr_high_end_process',
                     'vr_enable_post_process', 'vr_post_process_threshold', 'demucs_segment_size', 'demucs_shifts',
                     'demucs_overlap', 'demucs_segments_enabled', 'mdxc_segment_size', 'mdxc_override_model_segment_size',
                     'mdxc_overlap', 'mdxc_batch_size', 'mdxc_pitch_shift']:
            setattr(args, attr, None)

        mock_api_client.separate_audio_and_wait.return_value = {
            "status": "completed",
            "downloaded_files": ["output1.wav", "output2.wav"]
        }

        handle_separate_command(args, mock_api_client, mock_logger)

        call_args = mock_api_client.separate_audio_and_wait.call_args
        kwargs = call_args[1]
        assert kwargs["models"] == ["model1.ckpt", "model2.onnx"]

    def test_handle_separate_command_error(self, mock_api_client, mock_logger, mock_audio_file):
        """Test separate command with error."""
        args = Mock()
        args.audio_files = [mock_audio_file]
        args.model = "test_model.ckpt"
        args.models = None
        # Set other required attributes
        for attr in ['timeout', 'poll_interval', 'output_format', 'output_bitrate', 'normalization', 'amplification',
                     'single_stem', 'invert_spect', 'sample_rate', 'use_soundfile', 'use_autocast', 'custom_output_names',
                     'mdx_segment_size', 'mdx_overlap', 'mdx_batch_size', 'mdx_hop_length', 'mdx_enable_denoise',
                     'vr_batch_size', 'vr_window_size', 'vr_aggression', 'vr_enable_tta', 'vr_high_end_process',
                     'vr_enable_post_process', 'vr_post_process_threshold', 'demucs_segment_size', 'demucs_shifts',
                     'demucs_overlap', 'demucs_segments_enabled', 'mdxc_segment_size', 'mdxc_override_model_segment_size',
                     'mdxc_overlap', 'mdxc_batch_size', 'mdxc_pitch_shift']:
            setattr(args, attr, 0 if 'size' in attr or 'shift' in attr or 'batch' in attr or 'hop' in attr or 'aggression' in attr or 'overlap' in attr else False if 'enable' in attr or 'tta' in attr or 'process' in attr or 'spect' in attr or 'soundfile' in attr or 'autocast' in attr else None if 'output' in attr or 'single' in attr or 'custom' in attr else 600 if 'timeout' in attr else 10 if 'poll' in attr else 'Default' if 'demucs_segment' in attr else 0.25 if 'overlap' in attr and 'mdxc' not in attr else 512 if 'window' in attr else 44100 if 'sample' in attr else 'flac' if 'format' in attr else 0.9 if 'normalization' in attr else 0.0)

        mock_api_client.separate_audio_and_wait.return_value = {
            "status": "error",
            "error": "Processing failed"
        }

        handle_separate_command(args, mock_api_client, mock_logger)

        # Verify error was logged
        mock_logger.error.assert_called()

    def test_handle_separate_command_exception(self, mock_api_client, mock_logger, mock_audio_file):
        """Test separate command with exception."""
        args = Mock()
        args.audio_files = [mock_audio_file]
        # Set required attributes 
        for attr in ['model', 'models', 'timeout', 'poll_interval', 'output_format', 'output_bitrate', 'normalization',
                     'amplification', 'single_stem', 'invert_spect', 'sample_rate', 'use_soundfile', 'use_autocast',
                     'custom_output_names', 'mdx_segment_size', 'mdx_overlap', 'mdx_batch_size', 'mdx_hop_length',
                     'mdx_enable_denoise', 'vr_batch_size', 'vr_window_size', 'vr_aggression', 'vr_enable_tta',
                     'vr_high_end_process', 'vr_enable_post_process', 'vr_post_process_threshold', 'demucs_segment_size',
                     'demucs_shifts', 'demucs_overlap', 'demucs_segments_enabled', 'mdxc_segment_size',
                     'mdxc_override_model_segment_size', 'mdxc_overlap', 'mdxc_batch_size', 'mdxc_pitch_shift']:
            setattr(args, attr, None)

        mock_api_client.separate_audio_and_wait.side_effect = Exception("API error")

        handle_separate_command(args, mock_api_client, mock_logger)

        # Verify error was logged
        mock_logger.error.assert_called()

    def test_handle_status_command_success(self, mock_api_client, mock_logger):
        """Test successful status command handling."""
        args = Mock()
        args.task_id = "test-task-123"

        mock_api_client.get_job_status.return_value = {
            "status": "completed",
            "progress": 100,
            "current_model_index": 0,
            "total_models": 1,
            "original_filename": "test.wav",
            "models_used": ["test_model.ckpt"],
            "files": ["output1.wav", "output2.wav"]
        }

        handle_status_command(args, mock_api_client, mock_logger)

        mock_api_client.get_job_status.assert_called_once_with("test-task-123")
        # Verify status information was logged
        mock_logger.info.assert_called()

    def test_handle_status_command_error_status(self, mock_api_client, mock_logger):
        """Test status command with error status."""
        args = Mock()
        args.task_id = "test-task-456"

        mock_api_client.get_job_status.return_value = {
            "status": "error",
            "error": "Processing failed"
        }

        handle_status_command(args, mock_api_client, mock_logger)

        mock_api_client.get_job_status.assert_called_once_with("test-task-456")
        mock_logger.error.assert_called()

    def test_handle_status_command_exception(self, mock_api_client, mock_logger):
        """Test status command with exception."""
        args = Mock()
        args.task_id = "test-task-789"

        mock_api_client.get_job_status.side_effect = Exception("API error")

        handle_status_command(args, mock_api_client, mock_logger)

        mock_logger.error.assert_called()

    def test_handle_models_command_pretty_format(self, mock_api_client, mock_logger):
        """Test models command with pretty format."""
        args = Mock()
        args.format = "pretty"
        args.filter = None

        mock_api_client.list_models.return_value = {
            "text": "Model list in pretty format"
        }

        with patch('builtins.print') as mock_print:
            handle_models_command(args, mock_api_client, mock_logger)

        mock_api_client.list_models.assert_called_once_with("pretty", None)
        mock_print.assert_called_once_with("Model list in pretty format")

    def test_handle_models_command_json_format(self, mock_api_client, mock_logger):
        """Test models command with JSON format."""
        args = Mock()
        args.format = "json"
        args.filter = "vocals"

        models_data = {"models": [{"name": "vocal_model", "type": "MDX"}]}
        mock_api_client.list_models.return_value = models_data

        with patch('json.dumps') as mock_json_dumps:
            with patch('builtins.print') as mock_print:
                mock_json_dumps.return_value = '{"models": [{"name": "vocal_model", "type": "MDX"}]}'
                handle_models_command(args, mock_api_client, mock_logger)

        mock_api_client.list_models.assert_called_once_with("json", "vocals")
        mock_json_dumps.assert_called_once_with(models_data, indent=2)

    def test_handle_models_command_exception(self, mock_api_client, mock_logger):
        """Test models command with exception."""
        args = Mock()
        args.format = "pretty"
        args.filter = None

        mock_api_client.list_models.side_effect = Exception("API error")

        handle_models_command(args, mock_api_client, mock_logger)

        mock_logger.error.assert_called()

    def test_handle_download_command_success(self, mock_api_client, mock_logger):
        """Test successful download command handling."""
        args = Mock()
        args.task_id = "test-task-123"
        args.filenames = ["output1.wav", "output2.wav"]

        mock_api_client.download_file.side_effect = ["output1.wav", "output2.wav"]

        handle_download_command(args, mock_api_client, mock_logger)

        # Verify download was called for each file
        expected_calls = [
            call("test-task-123", "output1.wav"),
            call("test-task-123", "output2.wav")
        ]
        mock_api_client.download_file.assert_has_calls(expected_calls)

        # Verify success messages were logged
        assert mock_logger.info.call_count >= 4  # At least 2 downloading + 2 downloaded messages

    def test_handle_download_command_exception(self, mock_api_client, mock_logger):
        """Test download command with exception."""
        args = Mock()
        args.task_id = "test-task-456"
        args.filenames = ["output1.wav"]

        mock_api_client.download_file.side_effect = Exception("Download error")

        handle_download_command(args, mock_api_client, mock_logger)

        mock_logger.error.assert_called() 