#!/usr/bin/env python
import os
import logging
from typing import Optional

import requests


class AudioSeparatorAPIClient:
    """Client for interacting with a remotely deployed Audio Separator API."""

    def __init__(self, api_url: str, logger: logging.Logger):
        self.api_url = api_url
        self.logger = logger
        self.session = requests.Session()

    def separate_audio(self, file_path: str, model: Optional[str] = None) -> dict:
        """Submit audio separation job (asynchronous processing)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        files = {"file": (os.path.basename(file_path), open(file_path, "rb"))}
        data = {}

        if model:
            data["model"] = model

        try:
            # Increase timeout for large files (5 minutes)
            response = self.session.post(f"{self.api_url}/separate", files=files, data=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Separation request failed: {e}")
            raise
        finally:
            files["file"][1].close()

    def separate_audio_and_wait(self, file_path: str, model: Optional[str] = None, timeout: int = 600, poll_interval: int = 10, download: bool = True, output_dir: Optional[str] = None) -> dict:
        """
        Submit audio separation job and wait for completion (convenience method).

        This method handles the full workflow: submit job, poll for completion,
        and optionally download the result files.

        Args:
            file_path: Path to the audio file to separate
            model: Model to use for separation (optional)
            timeout: Maximum time to wait for completion in seconds (default: 600)
            poll_interval: How often to check status in seconds (default: 10)
            download: Whether to automatically download result files (default: True)
            output_dir: Directory to save downloaded files (default: current directory)

        Returns:
            dict with keys:
                - task_id: The job task ID
                - status: "completed" or "error"
                - files: List of output filenames
                - downloaded_files: List of local file paths (if download=True)
                - error: Error message (if status="error")
        """
        import time

        # Submit the separation job
        self.logger.info(f"Submitting separation job for '{file_path}'...")
        result = self.separate_audio(file_path, model)
        task_id = result["task_id"]
        self.logger.info(f"Job submitted! Task ID: {task_id}")

        # Poll for completion
        self.logger.info("Waiting for separation to complete...")
        start_time = time.time()
        last_progress = -1

        while time.time() - start_time < timeout:
            try:
                status = self.get_job_status(task_id)
                current_status = status.get("status", "unknown")

                # Show progress if it changed
                if "progress" in status and status["progress"] != last_progress:
                    self.logger.info(f"Progress: {status['progress']}%")
                    last_progress = status["progress"]

                # Check if completed
                if current_status == "completed":
                    self.logger.info("✅ Separation completed!")

                    result = {"task_id": task_id, "status": "completed", "files": status.get("files", [])}

                    # Download files if requested
                    if download:
                        downloaded_files = []
                        self.logger.info(f"📥 Downloading {len(status.get('files', []))} output files...")

                        for filename in status.get("files", []):
                            try:
                                if output_dir:
                                    output_path = f"{output_dir.rstrip('/')}/{filename}"
                                else:
                                    output_path = filename

                                downloaded_path = self.download_file(task_id, filename, output_path)
                                downloaded_files.append(downloaded_path)
                                self.logger.info(f"  ✅ Downloaded: {downloaded_path}")
                            except Exception as e:
                                self.logger.error(f"  ❌ Failed to download {filename}: {e}")

                        result["downloaded_files"] = downloaded_files
                        self.logger.info(f"🎉 Successfully downloaded {len(downloaded_files)} files!")

                    return result

                elif current_status == "error":
                    error_msg = status.get("error", "Unknown error")
                    self.logger.error(f"❌ Job failed: {error_msg}")
                    return {"task_id": task_id, "status": "error", "error": error_msg, "files": []}

                # Wait before next poll
                time.sleep(poll_interval)

            except Exception as e:
                self.logger.warning(f"Error polling status: {e}")
                time.sleep(poll_interval)

        # Timeout reached
        self.logger.error(f"❌ Job polling timed out after {timeout} seconds")
        return {"task_id": task_id, "status": "timeout", "error": f"Job polling timed out after {timeout} seconds", "files": []}

    def get_job_status(self, task_id: str) -> dict:
        """Get job status."""
        try:
            response = self.session.get(f"{self.api_url}/status/{task_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Status request failed: {e}")
            raise

    def download_file(self, task_id: str, filename: str, output_path: Optional[str] = None) -> str:
        """Download a file from a completed job."""
        if output_path is None:
            output_path = filename

        try:
            response = self.session.get(f"{self.api_url}/download/{task_id}/{filename}", timeout=60)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path
        except requests.RequestException as e:
            self.logger.error(f"Download failed: {e}")
            raise

    def list_models(self, format_type: str = "pretty", filter_by: Optional[str] = None) -> dict:
        """List available models."""
        try:
            if format_type == "json":
                response = self.session.get(f"{self.api_url}/models-json", timeout=10)
            else:
                url = f"{self.api_url}/models"
                if filter_by:
                    url += f"?filter_sort_by={filter_by}"
                response = self.session.get(url, timeout=10)

            response.raise_for_status()

            if format_type == "json":
                return response.json()
            else:
                return {"text": response.text}
        except requests.RequestException as e:
            self.logger.error(f"Models request failed: {e}")
            raise

    def get_server_version(self) -> str:
        """Get the server version."""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            response.raise_for_status()
            health_data = response.json()
            return health_data.get("version", "unknown")
        except requests.RequestException as e:
            self.logger.error(f"Health check request failed: {e}")
            raise
