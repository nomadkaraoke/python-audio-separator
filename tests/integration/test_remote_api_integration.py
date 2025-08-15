import json
import pytest
import logging
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse
import io

from audio_separator.remote import AudioSeparatorAPIClient


# Mock API Server for Integration Tests
class MockAPIHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for simulating the Audio Separator API."""

    # Class variables to store state across requests
    jobs = {}
    models_data = {
        "model1.ckpt": {"Type": "MDXC", "Stems": ["vocals (12.9)", "instrumental (17.0)"], "Name": "Test Model 1"},
        "model2.onnx": {"Type": "MDX", "Stems": ["vocals (10.5)", "instrumental (15.2)"], "Name": "Test Model 2"},
        "htdemucs_6s.yaml": {"Type": "Demucs", "Stems": ["vocals (9.7)", "drums (8.5)", "bass (10.0)", "guitar", "piano", "other"], "Name": "Test Demucs 6s"},
    }

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        path = self.path

        if path == "/health":
            self.send_json_response({"status": "healthy", "version": "1.0.0"})

        elif path.startswith("/status/"):
            task_id = path.split("/")[-1]
            if task_id in self.jobs:
                self.send_json_response(self.jobs[task_id])
            else:
                self.send_json_response({"task_id": task_id, "status": "not_found", "error": "Job not found"}, status=404)

        elif path.startswith("/download/"):
            # Parse task_id and filename from path like /download/task123/output.wav
            parts = path.split("/")
            if len(parts) >= 4:
                task_id = parts[2]
                filename = urllib.parse.unquote(parts[3])

                # Check if job exists and is completed
                if task_id in self.jobs and self.jobs[task_id]["status"] == "completed":
                    # Simulate audio file content
                    audio_content = b"fake audio file content for " + filename.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Disposition", f"attachment; filename={filename}")
                    self.end_headers()
                    self.wfile.write(audio_content)
                else:
                    self.send_json_response({"error": "File not found"}, status=404)
            else:
                self.send_json_response({"error": "Invalid download path"}, status=400)

        elif path == "/models" or path.startswith("/models?"):
            # Parse query parameters
            if "?" in path:
                query_string = path.split("?", 1)[1]
                params = urllib.parse.parse_qs(query_string)
                filter_by = params.get("filter_sort_by", [None])[0]
            else:
                filter_by = None

            # Filter models if requested
            filtered_models = self.models_data
            if filter_by:
                filtered_models = {k: v for k, v in self.models_data.items() if filter_by.lower() in " ".join(v["Stems"]).lower() or filter_by.lower() in v["Name"].lower()}

            # Format as plain text (like the real API)
            text_output = self._format_models_as_text(filtered_models)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(text_output.encode())

        elif path == "/models-json":
            self.send_json_response(self.models_data)

        else:
            self.send_json_response({"error": "Not found"}, status=404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/separate":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            # Simple task ID generation
            task_id = f"task-{len(self.jobs) + 1:03d}"

            # Initialize job status
            self.jobs[task_id] = {
                "task_id": task_id,
                "status": "submitted",
                "progress": 0,
                "original_filename": "test.wav",
                "models_used": ["default"],
                "total_models": 1,
                "current_model_index": 0,
                "files": [],
            }

            # Simulate processing in background (for testing polling)
            threading.Thread(target=self._simulate_processing, args=(task_id,)).start()

            self.send_json_response(
                {"task_id": task_id, "status": "submitted", "message": "Job submitted for processing", "models_used": ["default"], "total_models": 1, "original_filename": "test.wav"}
            )
        else:
            self.send_json_response({"error": "Not found"}, status=404)

    def _simulate_processing(self, task_id):
        """Simulate job processing in background."""
        time.sleep(0.1)  # Brief delay

        # Update to processing
        self.jobs[task_id].update({"status": "processing", "progress": 25})

        time.sleep(0.1)

        # Update progress
        self.jobs[task_id].update({"progress": 50})

        time.sleep(0.1)

        # Update progress
        self.jobs[task_id].update({"progress": 75})

        time.sleep(0.1)

        # Complete
        self.jobs[task_id].update({"status": "completed", "progress": 100, "files": ["test_(Vocals)_default.flac", "test_(Instrumental)_default.flac"]})

    def _format_models_as_text(self, models):
        """Format models dictionary as plain text table."""
        if not models:
            return "No models found"

        # Calculate column widths
        filename_width = max(len("Model Filename"), max(len(k) for k in models.keys()))
        arch_width = max(len("Arch"), max(len(v["Type"]) for v in models.values()))
        stems_width = max(len("Output Stems (SDR)"), max(len(", ".join(v["Stems"])) for v in models.values()))
        name_width = max(len("Friendly Name"), max(len(v["Name"]) for v in models.values()))

        total_width = filename_width + arch_width + stems_width + name_width + 15

        lines = []
        lines.append("-" * total_width)
        lines.append(f"{'Model Filename':<{filename_width}}  {'Arch':<{arch_width}}  {'Output Stems (SDR)':<{stems_width}}  {'Friendly Name'}")
        lines.append("-" * total_width)

        for filename, info in models.items():
            stems = ", ".join(info["Stems"])
            lines.append(f"{filename:<{filename_width}}  {info['Type']:<{arch_width}}  {stems:<{stems_width}}  {info['Name']}")

        return "\n".join(lines)

    def send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps(data).encode()
        self.wfile.write(response)


@pytest.fixture(scope="function")
def mock_http_server():
    """Start a mock HTTP server for testing."""
    server = HTTPServer(("localhost", 0), MockAPIHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Get the actual port the server is using
    port = server.server_address[1]
    base_url = f"http://localhost:{port}"

    yield base_url

    server.shutdown()
    server.server_close()


@pytest.fixture
def api_client(mock_http_server):
    """Create an API client connected to the mock server."""
    logger = logging.getLogger("test")
    return AudioSeparatorAPIClient(mock_http_server, logger)


@pytest.fixture
def test_audio_file():
    """Create a temporary test audio file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio content for testing")
        yield f.name
    os.unlink(f.name)


class TestRemoteAPIIntegration:
    """Integration tests for the Remote API functionality."""

    def test_server_health_check(self, api_client):
        """Test server health check endpoint."""
        version = api_client.get_server_version()
        assert version == "1.0.0"

    def test_list_models_pretty_format(self, api_client):
        """Test listing models in pretty format."""
        result = api_client.list_models(format_type="pretty")

        assert "text" in result
        text = result["text"]
        assert "Model Filename" in text
        assert "model1.ckpt" in text
        assert "model2.onnx" in text
        assert "htdemucs_6s.yaml" in text
        assert "Test Model 1" in text

    def test_list_models_json_format(self, api_client):
        """Test listing models in JSON format."""
        result = api_client.list_models(format_type="json")

        assert "model1.ckpt" in result
        assert "model2.onnx" in result
        assert "htdemucs_6s.yaml" in result
        assert result["model1.ckpt"]["Type"] == "MDXC"
        assert result["model2.onnx"]["Type"] == "MDX"
        assert result["htdemucs_6s.yaml"]["Type"] == "Demucs"

    def test_list_models_with_filter(self, api_client):
        """Test listing models with filter."""
        result = api_client.list_models(filter_by="vocals")

        assert "text" in result
        text = result["text"]
        # Should include models that have "vocals" in their stems
        assert "model1.ckpt" in text
        assert "model2.onnx" in text
        assert "htdemucs_6s.yaml" in text

    def test_separate_audio_submission(self, api_client, test_audio_file):
        """Test audio separation job submission."""
        result = api_client.separate_audio(test_audio_file)

        assert "task_id" in result
        assert result["status"] == "submitted"
        assert result["models_used"] == ["default"]
        assert result["total_models"] == 1
        assert result["original_filename"] == "test.wav"

        task_id = result["task_id"]
        assert task_id.startswith("task-")

    def test_separate_audio_with_custom_parameters(self, api_client, test_audio_file):
        """Test audio separation with custom parameters."""
        result = api_client.separate_audio(
            test_audio_file,
            model="model1.ckpt",
            output_format="wav",
            normalization_threshold=0.8,
            mdx_segment_size=512,
            vr_aggression=10,
            custom_output_names={"Vocals": "lead_vocals", "Instrumental": "backing_track"},
        )

        assert result["status"] == "submitted"
        assert "task_id" in result

    def test_job_status_polling(self, api_client, test_audio_file):
        """Test job status polling through completion."""
        # Submit job
        result = api_client.separate_audio(test_audio_file)
        task_id = result["task_id"]

        # Poll until completion
        max_attempts = 20  # Prevent infinite loop
        attempts = 0

        while attempts < max_attempts:
            status = api_client.get_job_status(task_id)

            assert status["task_id"] == task_id
            assert "status" in status

            if status["status"] == "completed":
                assert status["progress"] == 100
                assert "files" in status
                assert len(status["files"]) == 2  # Vocals and Instrumental
                break
            elif status["status"] in ["submitted", "processing"]:
                # Continue polling
                time.sleep(0.05)  # Small delay between polls
            else:
                pytest.fail(f"Unexpected job status: {status['status']}")

            attempts += 1

        if attempts >= max_attempts:
            pytest.fail("Job did not complete within expected time")

    def test_file_download(self, api_client, test_audio_file):
        """Test downloading files from completed job."""
        # Submit and wait for completion
        result = api_client.separate_audio(test_audio_file)
        task_id = result["task_id"]

        # Wait for completion (simplified polling)
        for _ in range(20):
            status = api_client.get_job_status(task_id)
            if status["status"] == "completed":
                break
            time.sleep(0.05)

        assert status["status"] == "completed"
        assert len(status["files"]) == 2

        # Download first file
        filename = status["files"][0]
        with tempfile.NamedTemporaryFile(delete=False) as temp_output:
            output_path = temp_output.name

        try:
            downloaded_path = api_client.download_file(task_id, filename, output_path)
            assert downloaded_path == output_path
            assert os.path.exists(output_path)

            # Verify file content
            with open(output_path, "rb") as f:
                content = f.read()
                assert content.startswith(b"fake audio file content")
                assert filename.encode() in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_separate_audio_and_wait_success(self, api_client, test_audio_file):
        """Test the convenience method for separating audio and waiting for completion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = api_client.separate_audio_and_wait(test_audio_file, timeout=5, poll_interval=0.05, download=True, output_dir=temp_dir)

            assert result["status"] == "completed"
            assert "task_id" in result
            assert "files" in result
            assert "downloaded_files" in result
            assert len(result["files"]) == 2
            assert len(result["downloaded_files"]) == 2

            # Verify files were actually downloaded
            for file_path in result["downloaded_files"]:
                assert os.path.exists(file_path)
                assert file_path.startswith(temp_dir)

    def test_separate_audio_and_wait_no_download(self, api_client, test_audio_file):
        """Test convenience method without downloading files."""
        result = api_client.separate_audio_and_wait(test_audio_file, timeout=5, poll_interval=0.05, download=False)

        assert result["status"] == "completed"
        assert "files" in result
        assert "downloaded_files" not in result

    def test_job_status_not_found(self, api_client):
        """Test getting status for non-existent job."""
        import requests

        with pytest.raises(requests.exceptions.HTTPError):
            api_client.get_job_status("nonexistent-task-id")

    def test_download_file_not_found(self, api_client):
        """Test downloading file for non-existent job."""
        with pytest.raises(Exception):  # Should raise an exception for 404
            api_client.download_file("nonexistent-task-id", "file.wav")

    def test_multiple_concurrent_jobs(self, api_client, test_audio_file):
        """Test handling multiple concurrent jobs."""
        # Submit multiple jobs
        num_jobs = 3
        jobs = []

        for i in range(num_jobs):
            result = api_client.separate_audio(test_audio_file)
            jobs.append(result["task_id"])

        # Wait for all jobs to complete
        completed_jobs = []
        max_attempts = 30

        for attempt in range(max_attempts):
            for task_id in jobs:
                if task_id not in completed_jobs:
                    status = api_client.get_job_status(task_id)
                    if status["status"] == "completed":
                        completed_jobs.append(task_id)

            if len(completed_jobs) == num_jobs:
                break

            time.sleep(0.05)

        assert len(completed_jobs) == num_jobs, "Not all jobs completed in expected time"

    def test_separate_audio_with_multiple_models(self, api_client, test_audio_file):
        """Test separation with multiple models (parameter passing)."""
        models = ["model1.ckpt", "model2.onnx"]
        result = api_client.separate_audio(test_audio_file, models=models)

        assert result["status"] == "submitted"
        assert "task_id" in result
        # Note: The mock server doesn't fully simulate multiple model processing,
        # but we can test that the parameters are accepted


# CLI Integration Tests (using the mock server)
class TestRemoteCLIIntegration:
    """Integration tests for the remote CLI."""

    @patch("audio_separator.remote.cli.AudioSeparatorAPIClient")
    def test_cli_separate_command_integration(self, mock_client_class, test_audio_file):
        """Test CLI separate command integration."""
        from audio_separator.remote.cli import handle_separate_command

        # Set up mock client
        mock_client = Mock()
        mock_client.separate_audio_and_wait.return_value = {"status": "completed", "downloaded_files": ["output1.wav", "output2.wav"]}
        mock_client_class.return_value = mock_client

        # Mock arguments (simplified)
        args = Mock()
        args.audio_files = [test_audio_file]
        args.model = "test_model.ckpt"
        args.models = None
        args.timeout = 600
        args.poll_interval = 10

        # Set all required attributes with appropriate defaults
        default_attrs = {
            "output_format": "flac",
            "output_bitrate": None,
            "normalization": 0.9,
            "amplification": 0.0,
            "single_stem": None,
            "invert_spect": False,
            "sample_rate": 44100,
            "use_soundfile": False,
            "use_autocast": False,
            "custom_output_names": None,
            "mdx_segment_size": 256,
            "mdx_overlap": 0.25,
            "mdx_batch_size": 1,
            "mdx_hop_length": 1024,
            "mdx_enable_denoise": False,
            "vr_batch_size": 1,
            "vr_window_size": 512,
            "vr_aggression": 5,
            "vr_enable_tta": False,
            "vr_high_end_process": False,
            "vr_enable_post_process": False,
            "vr_post_process_threshold": 0.2,
            "demucs_segment_size": "Default",
            "demucs_shifts": 2,
            "demucs_overlap": 0.25,
            "demucs_segments_enabled": True,
            "mdxc_segment_size": 256,
            "mdxc_override_model_segment_size": False,
            "mdxc_overlap": 8,
            "mdxc_batch_size": 1,
            "mdxc_pitch_shift": 0,
        }

        for attr, value in default_attrs.items():
            setattr(args, attr, value)

        logger = Mock()

        # Execute the command
        handle_separate_command(args, mock_client, logger)

        # Verify the API client method was called
        mock_client.separate_audio_and_wait.assert_called_once()

        # Verify success was logged
        logger.info.assert_called()


# End-to-End Test with Real Audio File
class TestRemoteAPIEndToEnd:
    """End-to-end tests using realistic audio data."""

    def test_end_to_end_workflow(self, api_client):
        """Test complete workflow from submission to download."""
        # Create a more realistic "audio" file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write some data that could represent a small WAV file
            f.write(b"RIFF")  # WAV header
            f.write(b"\x24\x00\x00\x00")  # File size
            f.write(b"WAVE")  # Format
            f.write(b"fake audio data for testing" * 10)  # Some fake audio data
            test_file = f.name

        try:
            # Step 1: Check server health
            version = api_client.get_server_version()
            assert version is not None

            # Step 2: List available models
            models = api_client.list_models()
            assert "text" in models

            # Step 3: Submit separation job
            result = api_client.separate_audio(test_file)
            task_id = result["task_id"]
            assert result["status"] == "submitted"

            # Step 4: Poll for completion
            completed = False
            for _ in range(20):
                status = api_client.get_job_status(task_id)
                if status["status"] == "completed":
                    completed = True
                    break
                elif status["status"] == "error":
                    pytest.fail(f"Job failed: {status.get('error', 'Unknown error')}")
                time.sleep(0.05)

            assert completed, "Job did not complete in expected time"

            # Step 5: Download results
            files = status["files"]
            assert len(files) > 0

            with tempfile.TemporaryDirectory() as temp_dir:
                for filename in files:
                    output_path = os.path.join(temp_dir, filename)
                    downloaded_path = api_client.download_file(task_id, filename, output_path)
                    assert os.path.exists(downloaded_path)
                    assert os.path.getsize(downloaded_path) > 0

        finally:
            os.unlink(test_file)

    def test_error_handling_workflow(self, api_client):
        """Test error handling in various scenarios."""
        import requests

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            api_client.separate_audio("/non/existent/file.wav")

        # Test status for non-existent job
        with pytest.raises(requests.exceptions.HTTPError):
            api_client.get_job_status("invalid-task-id")

        # Test download for non-existent job/file
        with pytest.raises(requests.exceptions.HTTPError):
            api_client.download_file("invalid-task-id", "file.wav")
