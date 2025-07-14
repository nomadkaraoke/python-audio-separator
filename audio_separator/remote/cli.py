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

    # Model selection
    model_group = separate_parser.add_mutually_exclusive_group()
    model_group.add_argument("-m", "--model", help="Single model to use for separation")
    model_group.add_argument("--models", nargs="+", help="Multiple models to use for separation")

    separate_parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for polling (default: 600)")
    separate_parser.add_argument("--poll_interval", type=int, default=10, help="Polling interval in seconds (default: 10)")

    # Output parameters
    output_group = separate_parser.add_argument_group("Output Parameters")
    output_group.add_argument("--output_format", default="flac", help="Output format for separated files (default: %(default)s)")
    output_group.add_argument("--output_bitrate", help="Output bitrate for separated files")
    output_group.add_argument("--normalization", type=float, default=0.9, help="Max peak amplitude to normalize audio to (default: %(default)s)")
    output_group.add_argument("--amplification", type=float, default=0.0, help="Min peak amplitude to amplify audio to (default: %(default)s)")
    output_group.add_argument("--single_stem", help="Output only single stem (e.g. Vocals, Instrumental)")
    output_group.add_argument("--invert_spect", action="store_true", help="Invert secondary stem using spectrogram")
    output_group.add_argument("--sample_rate", type=int, default=44100, help="Sample rate of output audio (default: %(default)s)")
    output_group.add_argument("--use_soundfile", action="store_true", help="Use soundfile for output writing")
    output_group.add_argument("--use_autocast", action="store_true", help="Use PyTorch autocast for faster inference")
    output_group.add_argument("--custom_output_names", type=json.loads, help="Custom output names in JSON format")

    # MDX parameters
    mdx_group = separate_parser.add_argument_group("MDX Architecture Parameters")
    mdx_group.add_argument("--mdx_segment_size", type=int, default=256, help="MDX segment size (default: %(default)s)")
    mdx_group.add_argument("--mdx_overlap", type=float, default=0.25, help="MDX overlap (default: %(default)s)")
    mdx_group.add_argument("--mdx_batch_size", type=int, default=1, help="MDX batch size (default: %(default)s)")
    mdx_group.add_argument("--mdx_hop_length", type=int, default=1024, help="MDX hop length (default: %(default)s)")
    mdx_group.add_argument("--mdx_enable_denoise", action="store_true", help="Enable MDX denoising")

    # VR parameters
    vr_group = separate_parser.add_argument_group("VR Architecture Parameters")
    vr_group.add_argument("--vr_batch_size", type=int, default=1, help="VR batch size (default: %(default)s)")
    vr_group.add_argument("--vr_window_size", type=int, default=512, help="VR window size (default: %(default)s)")
    vr_group.add_argument("--vr_aggression", type=int, default=5, help="VR aggression (default: %(default)s)")
    vr_group.add_argument("--vr_enable_tta", action="store_true", help="Enable VR Test-Time-Augmentation")
    vr_group.add_argument("--vr_high_end_process", action="store_true", help="Enable VR high end processing")
    vr_group.add_argument("--vr_enable_post_process", action="store_true", help="Enable VR post processing")
    vr_group.add_argument("--vr_post_process_threshold", type=float, default=0.2, help="VR post process threshold (default: %(default)s)")

    # Demucs parameters
    demucs_group = separate_parser.add_argument_group("Demucs Architecture Parameters")
    demucs_group.add_argument("--demucs_segment_size", default="Default", help="Demucs segment size (default: %(default)s)")
    demucs_group.add_argument("--demucs_shifts", type=int, default=2, help="Demucs shifts (default: %(default)s)")
    demucs_group.add_argument("--demucs_overlap", type=float, default=0.25, help="Demucs overlap (default: %(default)s)")
    demucs_group.add_argument("--demucs_segments_enabled", type=bool, default=True, help="Enable Demucs segments (default: %(default)s)")

    # MDXC parameters
    mdxc_group = separate_parser.add_argument_group("MDXC Architecture Parameters")
    mdxc_group.add_argument("--mdxc_segment_size", type=int, default=256, help="MDXC segment size (default: %(default)s)")
    mdxc_group.add_argument("--mdxc_override_model_segment_size", action="store_true", help="Override MDXC model segment size")
    mdxc_group.add_argument("--mdxc_overlap", type=int, default=8, help="MDXC overlap (default: %(default)s)")
    mdxc_group.add_argument("--mdxc_batch_size", type=int, default=1, help="MDXC batch size (default: %(default)s)")
    mdxc_group.add_argument("--mdxc_pitch_shift", type=int, default=0, help="MDXC pitch shift (default: %(default)s)")

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
            # Prepare parameters for separation
            kwargs = {
                "model": args.model,
                "models": args.models,
                "timeout": args.timeout,
                "poll_interval": args.poll_interval,
                "download": True,  # Always download in CLI
                "output_dir": None,  # Use current directory
                # Output parameters
                "output_format": args.output_format,
                "output_bitrate": args.output_bitrate,
                "normalization_threshold": args.normalization,
                "amplification_threshold": args.amplification,
                "output_single_stem": args.single_stem,
                "invert_using_spec": args.invert_spect,
                "sample_rate": args.sample_rate,
                "use_soundfile": args.use_soundfile,
                "use_autocast": args.use_autocast,
                "custom_output_names": args.custom_output_names,
                # MDX parameters
                "mdx_segment_size": args.mdx_segment_size,
                "mdx_overlap": args.mdx_overlap,
                "mdx_batch_size": args.mdx_batch_size,
                "mdx_hop_length": args.mdx_hop_length,
                "mdx_enable_denoise": args.mdx_enable_denoise,
                # VR parameters
                "vr_batch_size": args.vr_batch_size,
                "vr_window_size": args.vr_window_size,
                "vr_aggression": args.vr_aggression,
                "vr_enable_tta": args.vr_enable_tta,
                "vr_high_end_process": args.vr_high_end_process,
                "vr_enable_post_process": args.vr_enable_post_process,
                "vr_post_process_threshold": args.vr_post_process_threshold,
                # Demucs parameters
                "demucs_segment_size": args.demucs_segment_size,
                "demucs_shifts": args.demucs_shifts,
                "demucs_overlap": args.demucs_overlap,
                "demucs_segments_enabled": args.demucs_segments_enabled,
                # MDXC parameters
                "mdxc_segment_size": args.mdxc_segment_size,
                "mdxc_override_model_segment_size": args.mdxc_override_model_segment_size,
                "mdxc_overlap": args.mdxc_overlap,
                "mdxc_batch_size": args.mdxc_batch_size,
                "mdxc_pitch_shift": args.mdxc_pitch_shift,
            }

            # Use the convenience method that handles everything
            result = api_client.separate_audio_and_wait(audio_file, **kwargs)

            if result["status"] == "completed":
                if "downloaded_files" in result:
                    logger.info(f"✅ Separation completed! Downloaded {len(result['downloaded_files'])} files:")
                    for file_path in result["downloaded_files"]:
                        logger.info(f"  - {file_path}")
                else:
                    logger.info(f"✅ Separation completed! Files available for download: {result['files']}")
            else:
                logger.error(f"❌ Separation failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Error processing '{audio_file}': {e}")


def handle_status_command(args, api_client: AudioSeparatorAPIClient, logger: logging.Logger):
    """Handle the status command."""
    try:
        status = api_client.get_job_status(args.task_id)

        logger.info(f"Job Status: {status['status']}")
        if "progress" in status:
            progress_info = f"Progress: {status['progress']}%"
            if "current_model_index" in status and "total_models" in status:
                model_info = f" (Model {status['current_model_index'] + 1}/{status['total_models']})"
                progress_info += model_info
            logger.info(progress_info)
        if "original_filename" in status:
            logger.info(f"Original File: {status['original_filename']}")
        if "models_used" in status:
            logger.info(f"Models Used: {', '.join(status['models_used'])}")
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
                progress_info = f"📊 Progress: {status['progress']}%"
                if "current_model_index" in status and "total_models" in status:
                    model_info = f" (Model {status['current_model_index'] + 1}/{status['total_models']})"
                    progress_info += model_info
                logger.info(progress_info)
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
