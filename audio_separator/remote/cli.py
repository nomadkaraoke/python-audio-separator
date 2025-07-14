#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
import time
from importlib import metadata

from audio_separator.remote import AudioSeparatorAPIClient


def main():
    """Main entry point for the remote CLI."""
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(description="Separate audio files using a remote audio-separator API.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))

    # Get package version
    package_version = metadata.distribution("audio-separator").version

    # Main command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Separate command
    separate_parser = subparsers.add_parser("separate", help="Separate audio files")
    separate_parser.add_argument("audio_files", nargs="+", help="Audio file paths to separate")
    separate_parser.add_argument("-m", "--model", help="Model to use for separation")
    separate_parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for polling (default: 600)")
    separate_parser.add_argument("--poll_interval", type=int, default=10, help="Polling interval in seconds (default: 10)")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("task_id", help="Task ID to check status for")

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--format", choices=["pretty", "json"], default="pretty", help="Output format")
    models_parser.add_argument("--filter", help="Filter models by name, type, or stem")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download specific files from a job")
    download_parser.add_argument("task_id", help="Task ID")
    download_parser.add_argument("filenames", nargs="+", help="Filenames to download")

    # Global options
    parser.add_argument("-v", "--version", action="store_true", help="Show version information")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log_level", default="info", help="Log level (default: info)")
    parser.add_argument("--api_url", help="API URL (overrides AUDIO_SEPARATOR_API_URL env var)")

    args = parser.parse_args()

    # Set up logging
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    # Handle version command
    if args.version:
        print(f"Client version: {package_version}")

        # Try to get server version if API URL is available
        api_url = args.api_url or os.environ.get("AUDIO_SEPARATOR_API_URL")
        if api_url:
            api_url = api_url.rstrip("/")
            api_client = AudioSeparatorAPIClient(api_url, logger)
            try:
                server_version = api_client.get_server_version()
                print(f"Server version: {server_version}")
            except Exception as e:
                logger.warning(f"Could not retrieve server version: {e}")
        else:
            logger.warning("API URL not provided. Set AUDIO_SEPARATOR_API_URL environment variable or use --api_url to get server version")

        sys.exit(0)

    # Get API URL
    api_url = args.api_url or os.environ.get("AUDIO_SEPARATOR_API_URL")
    if not api_url:
        logger.error("API URL not provided. Set AUDIO_SEPARATOR_API_URL environment variable or use --api_url")
        sys.exit(1)

    # Remove trailing slash
    api_url = api_url.rstrip("/")

    # Create API client
    api_client = AudioSeparatorAPIClient(api_url, logger)

    # Handle commands
    if args.command == "separate":
        handle_separate_command(args, api_client, logger)
    elif args.command == "status":
        handle_status_command(args, api_client, logger)
    elif args.command == "models":
        handle_models_command(args, api_client, logger)
    elif args.command == "download":
        handle_download_command(args, api_client, logger)
    else:
        parser.print_help()
        sys.exit(1)


def handle_separate_command(args, api_client: AudioSeparatorAPIClient, logger: logging.Logger):
    """Handle the separate command."""
    for audio_file in args.audio_files:
        logger.info(f"Uploading '{audio_file}' to audio separator...")

        try:
            result = api_client.separate_audio(audio_file, args.model)

            # Poll for completion
            task_id = result["task_id"]
            logger.info(f"✅ Job submitted! Task ID: {task_id}")
            logger.info("🔄 Polling for completion...")

            if poll_for_completion(task_id, api_client, logger, args.timeout, args.poll_interval):
                # Get final status to get file list
                final_status = api_client.get_job_status(task_id)
                if final_status["status"] == "completed":
                    download_files(task_id, final_status["files"], api_client, logger)
                else:
                    logger.error(f"❌ Job failed: {final_status.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Error processing '{audio_file}': {e}")


def handle_status_command(args, api_client: AudioSeparatorAPIClient, logger: logging.Logger):
    """Handle the status command."""
    try:
        status = api_client.get_job_status(args.task_id)

        logger.info(f"Job Status: {status['status']}")
        if "progress" in status:
            logger.info(f"Progress: {status['progress']}%")
        if "original_filename" in status:
            logger.info(f"Original File: {status['original_filename']}")
        if "model_used" in status:
            logger.info(f"Model Used: {status['model_used']}")
        if status["status"] == "error" and "error" in status:
            logger.error(f"Error: {status['error']}")
        elif status["status"] == "completed" and "files" in status:
            logger.info("Output Files:")
            for filename in status["files"]:
                logger.info(f"  - {filename}")

    except Exception as e:
        logger.error(f"❌ Error getting status: {e}")


def handle_models_command(args, api_client: AudioSeparatorAPIClient, logger: logging.Logger):
    """Handle the models command."""
    try:
        models = api_client.list_models(args.format, args.filter)

        if args.format == "json":
            print(json.dumps(models, indent=2))
        else:
            print(models["text"])

    except Exception as e:
        logger.error(f"❌ Error listing models: {e}")


def handle_download_command(args, api_client: AudioSeparatorAPIClient, logger: logging.Logger):
    """Handle the download command."""
    try:
        for filename in args.filenames:
            logger.info(f"📂 Downloading: {filename}")
            output_path = api_client.download_file(args.task_id, filename)
            logger.info(f"✅ Downloaded: {output_path}")

    except Exception as e:
        logger.error(f"❌ Error downloading files: {e}")


def poll_for_completion(task_id: str, api_client: AudioSeparatorAPIClient, logger: logging.Logger, timeout: int = 600, poll_interval: int = 10) -> bool:
    """Poll for job completion."""
    start_time = time.time()
    last_progress = -1

    while time.time() - start_time < timeout:
        try:
            status = api_client.get_job_status(task_id)
            current_status = status.get("status", "unknown")

            # Show progress if it changed
            if "progress" in status and status["progress"] != last_progress:
                logger.info(f"📊 Progress: {status['progress']}%")
                last_progress = status["progress"]

            # Check if completed
            if current_status == "completed":
                logger.info("✅ Job completed!")
                return True
            elif current_status == "error":
                logger.error(f"❌ Job failed: {status.get('error', 'Unknown error')}")
                return False

            # Wait before next poll
            time.sleep(poll_interval)

        except Exception as e:
            logger.warning(f"Error polling status: {e}")
            time.sleep(poll_interval)

    logger.error(f"❌ Job polling timed out after {timeout} seconds")
    return False


def download_files(task_id: str, files: list, api_client: AudioSeparatorAPIClient, logger: logging.Logger):
    """Download all files from a completed job."""
    if not files:
        logger.warning("No files to download")
        return

    logger.info(f"📥 Downloading {len(files)} output files...")

    downloaded_count = 0
    for filename in files:
        try:
            logger.info(f"  📂 Downloading: {filename}")
            output_path = api_client.download_file(task_id, filename)
            logger.info(f"  ✅ Downloaded: {output_path}")
            downloaded_count += 1
        except Exception as e:
            logger.error(f"  ❌ Failed to download {filename}: {e}")

    if downloaded_count > 0:
        logger.info(f"🎉 Successfully downloaded {downloaded_count} files to current directory!")
    else:
        logger.error("❌ No files were successfully downloaded")


if __name__ == "__main__":
    main()
