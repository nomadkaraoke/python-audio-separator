#!/usr/bin/env python
import os
import logging
import json
from typing import Optional, List, Dict
from urllib.parse import quote

import requests

# Get package version for debugging
try:
    from importlib.metadata import version
    AUDIO_SEPARATOR_VERSION = version("audio-separator")
except ImportError:
    try:
        import pkg_resources
        AUDIO_SEPARATOR_VERSION = pkg_resources.get_distribution("audio-separator").version
    except Exception:
        AUDIO_SEPARATOR_VERSION = "unknown"


class AudioSeparatorAPIClient:
    """Client for interacting with a remotely deployed Audio Separator API."""

    def __init__(self, api_url: str, logger: logging.Logger):
        self.api_url = api_url
        self.logger = logger
        self.session = requests.Session()

    def separate_audio(
        self,
        file_path: str,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        # Output parameters
        output_format: str = "flac",
        output_bitrate: Optional[str] = None,
        normalization_threshold: float = 0.9,
        amplification_threshold: float = 0.0,
        output_single_stem: Optional[str] = None,
        invert_using_spec: bool = False,
        sample_rate: int = 44100,
        use_soundfile: bool = False,
        use_autocast: bool = False,
        custom_output_names: Optional[Dict[str, str]] = None,
        # MDX parameters
        mdx_segment_size: int = 256,
        mdx_overlap: float = 0.25,
        mdx_batch_size: int = 1,
        mdx_hop_length: int = 1024,
        mdx_enable_denoise: bool = False,
        # VR parameters
        vr_batch_size: int = 1,
        vr_window_size: int = 512,
        vr_aggression: int = 5,
        vr_enable_tta: bool = False,
        vr_high_end_process: bool = False,
        vr_enable_post_process: bool = False,
        vr_post_process_threshold: float = 0.2,
        # Demucs parameters
        demucs_segment_size: str = "Default",
        demucs_shifts: int = 2,
        demucs_overlap: float = 0.25,
        demucs_segments_enabled: bool = True,
        # MDXC parameters
        mdxc_segment_size: int = 256,
        mdxc_override_model_segment_size: bool = False,
        mdxc_overlap: int = 8,
        mdxc_batch_size: int = 1,
        mdxc_pitch_shift: int = 0,
    ) -> dict:
        """Submit audio separation job (asynchronous processing)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        files = {"file": (os.path.basename(file_path), open(file_path, "rb"))}
        data = {}

        # Handle model parameters (backwards compatibility)
        if models:
            data["models"] = json.dumps(models)
        elif model:
            data["model"] = model

        # Add all separator parameters
        data.update(
            {
                "output_format": output_format,
                "normalization_threshold": normalization_threshold,
                "amplification_threshold": amplification_threshold,
                "invert_using_spec": invert_using_spec,
                "sample_rate": sample_rate,
                "use_soundfile": use_soundfile,
                "use_autocast": use_autocast,
                # MDX parameters
                "mdx_segment_size": mdx_segment_size,
                "mdx_overlap": mdx_overlap,
                "mdx_batch_size": mdx_batch_size,
                "mdx_hop_length": mdx_hop_length,
                "mdx_enable_denoise": mdx_enable_denoise,
                # VR parameters
                "vr_batch_size": vr_batch_size,
                "vr_window_size": vr_window_size,
                "vr_aggression": vr_aggression,
                "vr_enable_tta": vr_enable_tta,
                "vr_high_end_process": vr_high_end_process,
                "vr_enable_post_process": vr_enable_post_process,
                "vr_post_process_threshold": vr_post_process_threshold,
                # Demucs parameters
                "demucs_segment_size": demucs_segment_size,
                "demucs_shifts": demucs_shifts,
                "demucs_overlap": demucs_overlap,
                "demucs_segments_enabled": demucs_segments_enabled,
                # MDXC parameters
                "mdxc_segment_size": mdxc_segment_size,
                "mdxc_override_model_segment_size": mdxc_override_model_segment_size,
                "mdxc_overlap": mdxc_overlap,
                "mdxc_batch_size": mdxc_batch_size,
                "mdxc_pitch_shift": mdxc_pitch_shift,
            }
        )

        # Add optional parameters only if they have non-default values
        if output_bitrate:
            data["output_bitrate"] = output_bitrate
        if output_single_stem:
            data["output_single_stem"] = output_single_stem
        if custom_output_names:
            data["custom_output_names"] = json.dumps(custom_output_names)

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

    def separate_audio_and_wait(
        self,
        file_path: str,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        download: bool = True,
        output_dir: Optional[str] = None,
        # All separator parameters (same as separate_audio)
        output_format: str = "flac",
        output_bitrate: Optional[str] = None,
        normalization_threshold: float = 0.9,
        amplification_threshold: float = 0.0,
        output_single_stem: Optional[str] = None,
        invert_using_spec: bool = False,
        sample_rate: int = 44100,
        use_soundfile: bool = False,
        use_autocast: bool = False,
        custom_output_names: Optional[Dict[str, str]] = None,
        mdx_segment_size: int = 256,
        mdx_overlap: float = 0.25,
        mdx_batch_size: int = 1,
        mdx_hop_length: int = 1024,
        mdx_enable_denoise: bool = False,
        vr_batch_size: int = 1,
        vr_window_size: int = 512,
        vr_aggression: int = 5,
        vr_enable_tta: bool = False,
        vr_high_end_process: bool = False,
        vr_enable_post_process: bool = False,
        vr_post_process_threshold: float = 0.2,
        demucs_segment_size: str = "Default",
        demucs_shifts: int = 2,
        demucs_overlap: float = 0.25,
        demucs_segments_enabled: bool = True,
        mdxc_segment_size: int = 256,
        mdxc_override_model_segment_size: bool = False,
        mdxc_overlap: int = 8,
        mdxc_batch_size: int = 1,
        mdxc_pitch_shift: int = 0,
    ) -> dict:
        """
        Submit audio separation job and wait for completion (convenience method).

        This method handles the full workflow: submit job, poll for completion,
        and optionally download the result files.

        Args:
            file_path: Path to the audio file to separate
            model: Single model to use for separation (for backwards compatibility)
            models: List of models to use for separation
            timeout: Maximum time to wait for completion in seconds (default: 600)
            poll_interval: How often to check status in seconds (default: 10)
            download: Whether to automatically download result files (default: True)
            output_dir: Directory to save downloaded files (default: current directory)
            **kwargs: All other separator parameters (same as separate_audio method)

        Returns:
            dict with keys:
                - task_id: The job task ID
                - status: "completed" or "error"
                - files: List of output filenames
                - downloaded_files: List of local file paths (if download=True)
                - error: Error message (if status="error")
        """
        import time

        # Submit the separation job with all parameters
        models_desc = models or ([model] if model else ["default"])
        self.logger.info(f"Submitting separation job for '{file_path}' with models: {models_desc} (audio-separator v{AUDIO_SEPARATOR_VERSION})")

        result = self.separate_audio(
            file_path,
            model,
            models,
            output_format,
            output_bitrate,
            normalization_threshold,
            amplification_threshold,
            output_single_stem,
            invert_using_spec,
            sample_rate,
            use_soundfile,
            use_autocast,
            custom_output_names,
            mdx_segment_size,
            mdx_overlap,
            mdx_batch_size,
            mdx_hop_length,
            mdx_enable_denoise,
            vr_batch_size,
            vr_window_size,
            vr_aggression,
            vr_enable_tta,
            vr_high_end_process,
            vr_enable_post_process,
            vr_post_process_threshold,
            demucs_segment_size,
            demucs_shifts,
            demucs_overlap,
            demucs_segments_enabled,
            mdxc_segment_size,
            mdxc_override_model_segment_size,
            mdxc_overlap,
            mdxc_batch_size,
            mdxc_pitch_shift,
        )

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
                    progress_info = f"Progress: {status['progress']}%"
                    if "current_model_index" in status and "total_models" in status:
                        model_info = f" (Model {status['current_model_index'] + 1}/{status['total_models']})"
                        progress_info += model_info
                    self.logger.info(progress_info)
                    last_progress = status["progress"]

                # Check if completed
                if current_status == "completed":
                    self.logger.info("✅ Separation completed!")
                    
                    files_data = status.get("files", {})
                    
                    # Handle both old (list) and new (dict) format for backward compatibility
                    if isinstance(files_data, list):
                        # Legacy format: list of filenames
                        self.logger.info(f"🔍 Job status returned {len(files_data)} files (legacy format)")
                        for i, filename in enumerate(files_data):
                            self.logger.info(f"  [{i}] '{filename}' (len={len(filename)})")
                        result = {"task_id": task_id, "status": "completed", "files": files_data}
                    else:
                        # New format: dictionary of hash -> filename
                        self.logger.info(f"🔍 Job status returned {len(files_data)} files (hash format)")
                        for i, (file_hash, filename) in enumerate(files_data.items()):
                            self.logger.info(f"  [{i}] hash={file_hash} -> '{filename}' (len={len(filename)})")
                        result = {"task_id": task_id, "status": "completed", "files": files_data}

                    # Download files if requested
                    if download:
                        downloaded_files = []
                        files_data = status.get("files", {})
                        
                        # Handle both old (list) and new (dict) format
                        if isinstance(files_data, list):
                            # Legacy format: list of filenames
                            self.logger.info(f"📥 Downloading {len(files_data)} output files (legacy format)...")
                            self.logger.info(f"🔍 Files to download: {files_data}")

                            for i, filename in enumerate(files_data):
                                try:
                                    self.logger.info(f"🔍 [{i+1}/{len(files_data)}] Attempting to download: '{filename}' (len={len(filename)})")
                                    
                                    if output_dir:
                                        output_path = f"{output_dir.rstrip('/')}/{filename}"
                                    else:
                                        output_path = filename

                                    downloaded_path = self.download_file(task_id, filename, output_path)
                                    downloaded_files.append(downloaded_path)
                                    self.logger.info(f"  ✅ Downloaded: {downloaded_path}")
                                except Exception as e:
                                    self.logger.error(f"  ❌ Failed to download {filename}: {e}")
                                    self._log_server_version_on_error()
                        else:
                            # New format: dictionary of hash -> filename
                            self.logger.info(f"📥 Downloading {len(files_data)} output files (hash format)...")
                            filenames_list = list(files_data.values())
                            self.logger.info(f"🔍 Files to download: {filenames_list}")

                            for i, (file_hash, filename) in enumerate(files_data.items()):
                                try:
                                    self.logger.info(f"🔍 [{i+1}/{len(files_data)}] Attempting to download: '{filename}' (hash={file_hash}, len={len(filename)})")
                                    
                                    if output_dir:
                                        output_path = f"{output_dir.rstrip('/')}/{filename}"
                                    else:
                                        output_path = filename

                                    downloaded_path = self.download_file_by_hash(task_id, file_hash, filename, output_path)
                                    downloaded_files.append(downloaded_path)
                                    self.logger.info(f"  ✅ Downloaded: {downloaded_path}")
                                except Exception as e:
                                    self.logger.error(f"  ❌ Failed to download {filename}: {e}")
                                    self._log_server_version_on_error()

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
        """Download a file from a completed job (legacy method for backward compatibility)."""
        if output_path is None:
            output_path = filename

        try:
            # URL encode the filename to handle spaces and special characters
            encoded_filename = quote(filename, safe='')
            download_url = f"{self.api_url}/download/{task_id}/{encoded_filename}"
            
            # Debug logging to understand what's happening
            self.logger.info(f"🔍 Download details (legacy filename method):")
            self.logger.info(f"  Original filename: '{filename}'")
            self.logger.info(f"  Encoded filename: '{encoded_filename}'")
            self.logger.info(f"  Download URL: {download_url}")
            self.logger.info(f"  Task ID: {task_id}")
            
            response = self.session.get(download_url, timeout=60)
            
            # Log response details for debugging
            self.logger.info(f"🔍 Response status: {response.status_code}")
            if response.status_code != 200:
                try:
                    self.logger.error(f"🔍 Response headers: {dict(response.headers)}")
                except Exception:
                    self.logger.error(f"🔍 Response headers: {response.headers}")
                try:
                    self.logger.error(f"🔍 Response text (first 500 chars): {response.text[:500]}")
                except Exception:
                    self.logger.error(f"🔍 Response text: <unavailable>")
            
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path
        except requests.RequestException as e:
            self.logger.error(f"Download failed: {e}")
            raise

    def download_file_by_hash(self, task_id: str, file_hash: str, filename: str, output_path: Optional[str] = None) -> str:
        """Download a file from a completed job using its hash identifier."""
        if output_path is None:
            output_path = filename

        try:
            # Use the file hash in the URL instead of the filename
            download_url = f"{self.api_url}/download/{task_id}/{file_hash}"
            
            # Debug logging to understand what's happening
            self.logger.info(f"🔍 Download details (hash method):")
            self.logger.info(f"  Original filename: '{filename}'")
            self.logger.info(f"  File hash: '{file_hash}'")
            self.logger.info(f"  Download URL: {download_url}")
            self.logger.info(f"  Task ID: {task_id}")
            
            response = self.session.get(download_url, timeout=60)
            
            # Log response details for debugging
            self.logger.info(f"🔍 Response status: {response.status_code}")
            if response.status_code != 200:
                try:
                    self.logger.error(f"🔍 Response headers: {dict(response.headers)}")
                except Exception:
                    self.logger.error(f"🔍 Response headers: {response.headers}")
                try:
                    self.logger.error(f"🔍 Response text (first 500 chars): {response.text[:500]}")
                except Exception:
                    self.logger.error(f"🔍 Response text: <unavailable>")
            
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path
        except requests.RequestException as e:
            self.logger.error(f"Download failed: {e}")
            raise

    def _log_server_version_on_error(self):
        """Helper method to log server version when download fails."""
        try:
            server_version = self.get_server_version()
            self.logger.error(f"🔍 Server version when download failed: {server_version}")
        except Exception as version_error:
            self.logger.error(f"🔍 Could not get server version: {version_error}")

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
