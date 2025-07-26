import json
import pytest
import logging
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from unittest.mock import MagicMock
import requests

from audio_separator.remote import AudioSeparatorAPIClient


@pytest.fixture
def logger():
    """Create a mock logger for testing."""
    return logging.getLogger("test")


@pytest.fixture
def api_client(logger):
    """Create an API client instance for testing."""
    return AudioSeparatorAPIClient("https://test-api.example.com", logger)


@pytest.fixture
def mock_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio content")
        yield f.name
    os.unlink(f.name)


class TestAudioSeparatorAPIClient:
    """Test the AudioSeparatorAPIClient class."""

    def test_init(self, logger):
        """Test client initialization."""
        api_url = "https://test-api.example.com"
        client = AudioSeparatorAPIClient(api_url, logger)

        assert client.api_url == api_url
        assert client.logger == logger
        assert client.session is not None

    @patch("requests.Session.post")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake audio content")
    def test_separate_audio_success(self, mock_file, mock_post, api_client, mock_audio_file):
        """Test successful audio separation submission."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "task_id": "test-task-123",
            "status": "submitted",
            "message": "Job submitted for processing",
            "models_used": ["default"],
            "total_models": 1,
            "original_filename": "test.wav",
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = api_client.separate_audio(mock_audio_file)

        # Verify the result
        assert result["task_id"] == "test-task-123"
        assert result["status"] == "submitted"
        assert result["models_used"] == ["default"]

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://test-api.example.com/separate"
        assert "files" in call_args[1]
        assert "data" in call_args[1]

    @patch("requests.Session.post")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake audio content")
    def test_separate_audio_with_multiple_models(self, mock_file, mock_post, api_client, mock_audio_file):
        """Test audio separation with multiple models."""
        mock_response = Mock()
        mock_response.json.return_value = {"task_id": "test-task-456", "status": "submitted", "models_used": ["model1.ckpt", "model2.onnx"], "total_models": 2}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        models = ["model1.ckpt", "model2.onnx"]
        result = api_client.separate_audio(mock_audio_file, models=models)

        assert result["models_used"] == models
        assert result["total_models"] == 2

        # Check that models were serialized correctly in the request
        call_args = mock_post.call_args
        data = call_args[1]["data"]
        assert json.loads(data["models"]) == models

    @patch("requests.Session.post")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake audio content")
    def test_separate_audio_with_custom_parameters(self, mock_file, mock_post, api_client, mock_audio_file):
        """Test audio separation with custom parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"task_id": "test-task-789", "status": "submitted"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        custom_output_names = {"Vocals": "lead_vocals", "Instrumental": "backing_track"}
        result = api_client.separate_audio(
            mock_audio_file, model="test_model.ckpt", output_format="wav", normalization_threshold=0.8, mdx_segment_size=512, vr_aggression=10, custom_output_names=custom_output_names
        )

        # Verify the parameters were passed correctly
        call_args = mock_post.call_args
        data = call_args[1]["data"]
        assert data["model"] == "test_model.ckpt"
        assert data["output_format"] == "wav"
        assert data["normalization_threshold"] == 0.8
        assert data["mdx_segment_size"] == 512
        assert data["vr_aggression"] == 10
        assert json.loads(data["custom_output_names"]) == custom_output_names

    @patch("requests.Session.post")
    def test_separate_audio_file_not_found(self, mock_post, api_client):
        """Test audio separation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            api_client.separate_audio("/nonexistent/file.wav")

    @patch("requests.Session.post")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake audio content")
    def test_separate_audio_request_error(self, mock_file, mock_post, api_client, mock_audio_file):
        """Test audio separation with request error."""
        mock_post.side_effect = requests.RequestException("Connection error")

        with pytest.raises(requests.RequestException):
            api_client.separate_audio(mock_audio_file)

    @patch("requests.Session.get")
    def test_get_job_status_success(self, mock_get, api_client):
        """Test successful job status retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"task_id": "test-task-123", "status": "processing", "progress": 50, "current_model_index": 0, "total_models": 1}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_job_status("test-task-123")

        assert result["status"] == "processing"
        assert result["progress"] == 50
        mock_get.assert_called_once_with("https://test-api.example.com/status/test-task-123", timeout=10)

    @patch("requests.Session.get")
    def test_get_job_status_error(self, mock_get, api_client):
        """Test job status retrieval with error."""
        mock_get.side_effect = requests.RequestException("API error")

        with pytest.raises(requests.RequestException):
            api_client.get_job_status("test-task-123")

    @patch("requests.Session.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_success(self, mock_file, mock_get, api_client):
        """Test successful file download."""
        mock_response = Mock()
        mock_response.content = b"fake audio file content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.download_file("test-task-123", "output.wav", "local_output.wav")

        assert result == "local_output.wav"
        mock_get.assert_called_once_with("https://test-api.example.com/download/test-task-123/output.wav", timeout=60)
        mock_file.assert_called_once_with("local_output.wav", "wb")

    @patch("requests.Session.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_default_output_path(self, mock_file, mock_get, api_client):
        """Test file download with default output path."""
        mock_response = Mock()
        mock_response.content = b"fake audio file content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.download_file("test-task-123", "output.wav")

        assert result == "output.wav"
        mock_file.assert_called_once_with("output.wav", "wb")

    @patch("requests.Session.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_with_spaces_in_filename(self, mock_file, mock_get, api_client):
        """Test file download with spaces in filename (URL encoding)."""
        mock_response = Mock()
        mock_response.content = b"fake audio file content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        filename_with_spaces = "My Song (Vocals) Output.wav"
        result = api_client.download_file("test-task-123", filename_with_spaces)

        # Verify URL was properly encoded
        expected_url = "https://test-api.example.com/download/test-task-123/My%20Song%20%28Vocals%29%20Output.wav"
        mock_get.assert_called_once_with(expected_url, timeout=60)
        assert result == filename_with_spaces
        mock_file.assert_called_once_with(filename_with_spaces, "wb")

    @patch("requests.Session.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_with_special_characters(self, mock_file, mock_get, api_client):
        """Test file download with special characters in filename."""
        mock_response = Mock()
        mock_response.content = b"fake audio file content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        filename_with_special_chars = "Song & Band - Title (Vocals 50% Mix).flac"
        result = api_client.download_file("test-task-456", filename_with_special_chars)

        # Verify URL was properly encoded
        expected_url = "https://test-api.example.com/download/test-task-456/Song%20%26%20Band%20-%20Title%20%28Vocals%2050%25%20Mix%29.flac"
        mock_get.assert_called_once_with(expected_url, timeout=60)
        assert result == filename_with_special_chars
        mock_file.assert_called_once_with(filename_with_special_chars, "wb")

    @patch("requests.Session.get")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_with_unicode_characters(self, mock_file, mock_get, api_client):
        """Test file download with unicode characters in filename."""
        mock_response = Mock()
        mock_response.content = b"fake audio file content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        unicode_filename = "Café - Naïve Song (Résumé).mp3"
        result = api_client.download_file("test-task-789", unicode_filename)

        # Verify URL was properly encoded (UTF-8 encoded then percent-encoded)
        expected_url = "https://test-api.example.com/download/test-task-789/Caf%C3%A9%20-%20Na%C3%AFve%20Song%20%28R%C3%A9sum%C3%A9%29.mp3"
        mock_get.assert_called_once_with(expected_url, timeout=60)
        assert result == unicode_filename
        mock_file.assert_called_once_with(unicode_filename, "wb")

    @patch("requests.Session.get")
    def test_download_file_error(self, mock_get, api_client):
        """Test file download with error."""
        mock_get.side_effect = requests.RequestException("Download error")

        with pytest.raises(requests.RequestException):
            api_client.download_file("test-task-123", "output.wav")

    @patch("requests.Session.get")
    def test_list_models_pretty_format(self, mock_get, api_client):
        """Test listing models in pretty format."""
        mock_response = Mock()
        mock_response.text = "Model list in pretty format"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.list_models(format_type="pretty")

        assert result == {"text": "Model list in pretty format"}
        mock_get.assert_called_once_with("https://test-api.example.com/models", timeout=10)

    @patch("requests.Session.get")
    def test_list_models_json_format(self, mock_get, api_client):
        """Test listing models in JSON format."""
        mock_response = Mock()
        models_data = {"models": [{"name": "model1", "type": "MDX"}]}
        mock_response.json.return_value = models_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.list_models(format_type="json")

        assert result == models_data
        mock_get.assert_called_once_with("https://test-api.example.com/models-json", timeout=10)

    @patch("requests.Session.get")
    def test_list_models_with_filter(self, mock_get, api_client):
        """Test listing models with filter."""
        mock_response = Mock()
        mock_response.text = "Filtered model list"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.list_models(filter_by="vocals")

        assert result == {"text": "Filtered model list"}
        mock_get.assert_called_once_with("https://test-api.example.com/models?filter_sort_by=vocals", timeout=10)

    @patch("requests.Session.get")
    def test_get_server_version_success(self, mock_get, api_client):
        """Test successful server version retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"version": "1.2.3", "status": "healthy"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_server_version()

        assert result == "1.2.3"
        mock_get.assert_called_once_with("https://test-api.example.com/health", timeout=10)

    @patch("requests.Session.get")
    def test_get_server_version_no_version(self, mock_get, api_client):
        """Test server version retrieval when version is not in response."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_server_version()

        assert result == "unknown"

    @patch("requests.Session.get")
    def test_get_server_version_error(self, mock_get, api_client):
        """Test server version retrieval with error."""
        mock_get.side_effect = requests.RequestException("Health check failed")

        with pytest.raises(requests.RequestException):
            api_client.get_server_version()

    @patch.object(AudioSeparatorAPIClient, "separate_audio")
    @patch.object(AudioSeparatorAPIClient, "get_job_status")
    @patch.object(AudioSeparatorAPIClient, "download_file")
    @patch("time.sleep")
    def test_separate_audio_and_wait_success(self, mock_sleep, mock_download, mock_status, mock_separate, api_client, mock_audio_file):
        """Test the convenience method for separating audio and waiting for completion."""
        # Mock separation submission
        mock_separate.return_value = {"task_id": "test-task-123"}

        # Mock status polling - first processing, then completed
        mock_status.side_effect = [{"status": "processing", "progress": 25}, {"status": "processing", "progress": 50}, {"status": "completed", "files": ["output1.wav", "output2.wav"]}]

        # Mock file downloads
        mock_download.side_effect = ["output1.wav", "output2.wav"]

        result = api_client.separate_audio_and_wait(mock_audio_file, model="test_model.ckpt", timeout=60, poll_interval=5, download=True)

        # Verify the result
        assert result["status"] == "completed"
        assert result["task_id"] == "test-task-123"
        assert result["files"] == ["output1.wav", "output2.wav"]
        assert result["downloaded_files"] == ["output1.wav", "output2.wav"]

        # Verify method calls
        mock_separate.assert_called_once()
        assert mock_status.call_count == 3
        assert mock_download.call_count == 2
        assert mock_sleep.call_count == 2  # Two polling iterations

    @patch.object(AudioSeparatorAPIClient, "separate_audio")
    @patch.object(AudioSeparatorAPIClient, "get_job_status")
    @patch("time.sleep")
    def test_separate_audio_and_wait_error(self, mock_sleep, mock_status, mock_separate, api_client, mock_audio_file):
        """Test the convenience method when job fails."""
        mock_separate.return_value = {"task_id": "test-task-456"}
        mock_status.side_effect = [{"status": "processing", "progress": 25}, {"status": "error", "error": "Processing failed"}]

        result = api_client.separate_audio_and_wait(mock_audio_file, timeout=60, poll_interval=5)

        assert result["status"] == "error"
        assert result["error"] == "Processing failed"
        assert result["files"] == []

    @patch.object(AudioSeparatorAPIClient, "separate_audio")
    @patch.object(AudioSeparatorAPIClient, "get_job_status")
    @patch("time.sleep")
    def test_separate_audio_and_wait_timeout(self, mock_sleep, mock_status, mock_separate, api_client, mock_audio_file):
        """Test the convenience method when polling times out."""
        mock_separate.return_value = {"task_id": "test-task-789"}
        mock_status.return_value = {"status": "processing", "progress": 25}

        result = api_client.separate_audio_and_wait(mock_audio_file, timeout=1, poll_interval=0.1)

        assert result["status"] == "timeout"
        assert "timed out" in result["error"]

    @patch.object(AudioSeparatorAPIClient, "separate_audio")
    @patch.object(AudioSeparatorAPIClient, "get_job_status")
    @patch.object(AudioSeparatorAPIClient, "download_file")
    @patch("time.sleep")
    def test_separate_audio_and_wait_no_download(self, mock_sleep, mock_download, mock_status, mock_separate, api_client, mock_audio_file):
        """Test the convenience method without downloading files."""
        mock_separate.return_value = {"task_id": "test-task-123"}
        mock_status.side_effect = [{"status": "completed", "files": ["output1.wav", "output2.wav"]}]

        result = api_client.separate_audio_and_wait(mock_audio_file, download=False)

        assert result["status"] == "completed"
        assert "downloaded_files" not in result
        mock_download.assert_not_called()

    @patch.object(AudioSeparatorAPIClient, "separate_audio")
    @patch.object(AudioSeparatorAPIClient, "get_job_status")
    @patch.object(AudioSeparatorAPIClient, "download_file")
    @patch("time.sleep")
    def test_separate_audio_and_wait_with_output_dir(self, mock_sleep, mock_download, mock_status, mock_separate, api_client, mock_audio_file):
        """Test the convenience method with custom output directory."""
        mock_separate.return_value = {"task_id": "test-task-123"}
        mock_status.side_effect = [{"status": "completed", "files": ["output1.wav"]}]
        mock_download.return_value = "custom_dir/output1.wav"

        result = api_client.separate_audio_and_wait(mock_audio_file, download=True, output_dir="custom_dir")

        # Verify download was called with custom output path
        mock_download.assert_called_once_with("test-task-123", "output1.wav", "custom_dir/output1.wav")
        assert result["downloaded_files"] == ["custom_dir/output1.wav"]

    @patch.object(AudioSeparatorAPIClient, "separate_audio")
    @patch.object(AudioSeparatorAPIClient, "get_job_status")
    @patch.object(AudioSeparatorAPIClient, "download_file")
    @patch("time.sleep")
    def test_separate_audio_and_wait_with_special_character_filenames(self, mock_sleep, mock_download, mock_status, mock_separate, api_client, mock_audio_file):
        """Test the convenience method with filenames containing special characters."""
        mock_separate.return_value = {"task_id": "test-task-456"}
        
        # Simulate files with special characters like those in the bug report
        files_with_special_chars = [
            "Song (Vocals model_bs_roformer_ep_317_sdr_12.9755.ckpt)_(Instrumental)_mel_band_roformer.flac",
            "Song (Vocals model_bs_roformer_ep_317_sdr_12.9755.ckpt)_(Vocals)_mel_band_roformer.flac"
        ]
        mock_status.side_effect = [{"status": "completed", "files": files_with_special_chars}]
        
        # Mock successful downloads
        mock_download.side_effect = files_with_special_chars

        result = api_client.separate_audio_and_wait(mock_audio_file, download=True)

        # Verify both files were downloaded despite having special characters
        assert result["status"] == "completed"
        assert result["downloaded_files"] == files_with_special_chars
        assert mock_download.call_count == 2
        
        # Verify download was called with the correct filenames
        expected_calls = [
            ("test-task-456", files_with_special_chars[0], files_with_special_chars[0]),
            ("test-task-456", files_with_special_chars[1], files_with_special_chars[1])
        ]
        actual_calls = [call.args for call in mock_download.call_args_list]
        assert actual_calls == expected_calls
