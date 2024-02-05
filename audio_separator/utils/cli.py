#!/usr/bin/env python
import argparse
import logging
import json
from importlib import metadata


def main():
    """Main entry point for the CLI."""

    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(description="Separate audio file into different stems.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))

    parser.add_argument("audio_file", nargs="?", help="The audio file path to separate, in any common format.", default=argparse.SUPPRESS)

    package_version = metadata.distribution("audio-separator").version

    info_params = parser.add_argument_group("Info and Debugging")
    info_params.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}")
    info_params.add_argument("-d", "--debug", action="store_true", help="enable debug logging, equivalent to --log_level=debug")
    info_params.add_argument("-e", "--env_info", action="store_true", help="print environment information and exit.")
    info_params.add_argument("-m", "--list_models", action="store_true", help="list all supported models and exit.")
    info_params.add_argument("--log_level", default="info", help="log level, e.g. info, debug, warning (default: %(default)s)")

    io_params = parser.add_argument_group("Separation I/O Params")
    io_params.add_argument("--model_filename", default="UVR-MDX-NET-Inst_HQ_3.onnx", help="model to use for separation (default: %(default)s). Example: --model_filename=2_HP-UVR.pth")
    io_params.add_argument("--output_format", default="FLAC", help="output format for separated files, any common format (default: %(default)s). Example: --output_format=MP3")
    io_params.add_argument("--output_dir", default=None, help="directory to write output files (default: <current dir>). Example: --output_dir=/app/separated")
    io_params.add_argument("--model_file_dir", default="/tmp/audio-separator-models/", help="model files directory (default: %(default)s). Example: --model_file_dir=/app/models")

    common_params = parser.add_argument_group("Common Separation Parameters")
    common_params.add_argument("--denoise", action="store_true", help="enable denoising during separation (default: %(default)s). Example: --denoise")
    common_params.add_argument("--invert_spect", action="store_true", help="invert secondary stem using spectogram (default: %(default)s). Example: --invert_spect")
    common_params.add_argument("--normalization", type=float, default=0.9, help="max peak amplitude to normalize input and output audio to (default: %(default)s). Example: --normalization=0.7")
    common_params.add_argument("--single_stem", default=None, help="output only single stem, either instrumental or vocals. Example: --single_stem=instrumental")
    common_params.add_argument("--sample_rate", type=int, default=44100, help="modify the sample rate of the output audio (default: %(default)s). Example: --sample_rate=44100")

    mdx_params = parser.add_argument_group("MDX Architecture Parameters")
    mdx_params.add_argument("--mdx_segment_size", type=int, default=256, help="larger consumes more resources, but may give better results (default: %(default)s). Example: --mdx_segment_size=256")
    mdx_params.add_argument(
        "--mdx_overlap", type=float, default=0.25, help="amount of overlap between prediction windows, 0.001-0.999. higher is better but slower (default: %(default)s). Example: --mdx_overlap=0.25"
    )
    mdx_params.add_argument("--mdx_batch_size", type=int, default=1, help="larger consumes more RAM but may process slightly faster (default: %(default)s). Example: --mdx_batch_size=4")
    mdx_params.add_argument(
        "--mdx_hop_length", type=int, default=1024, help="usually called stride in neural networks, only change if you know what you're doing (default: %(default)s). Example: --mdx_hop_length=1024"
    )

    vr_params = parser.add_argument_group("VR Architecture Parameters")
    vr_params.add_argument(
        "--vr_batch_size", type=int, default=4, help="number of batches to process at a time. higher = more RAM, slightly faster processing (default: %(default)s). Example: --vr_batch_size=16"
    )
    vr_params.add_argument(
        "--vr_window_size", type=int, default=512, help="balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: %(default)s). Example: --vr_window_size=320"
    )
    vr_params.add_argument(
        "--vr_aggression", type=int, default=5, help="intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: %(default)s). Example: --vr_aggression=2"
    )
    vr_params.add_argument("--vr_enable_tta", action="store_true", help="enable Test-Time-Augmentation; slow but improves quality (default: %(default)s). Example: --vr_enable_tta")
    vr_params.add_argument("--vr_high_end_process", action="store_true", help="mirror the missing frequency range of the output (default: %(default)s). Example: --vr_high_end_process")
    vr_params.add_argument(
        "--vr_enable_post_process",
        action="store_true",
        help="identify leftover artifacts within vocal output; may improve separation for some songs (default: %(default)s). Example: --vr_enable_post_process",
    )
    vr_params.add_argument("--vr_post_process_threshold", type=float, default=0.2, help="threshold for post_process feature: 0.1-0.3 (default: %(default)s). Example: --vr_post_process_threshold=0.1")

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper())

    logger.setLevel(log_level)

    if args.env_info:
        from audio_separator.separator import Separator

        separator = Separator()
        exit(0)

    if args.list_models:
        from audio_separator.separator import Separator

        separator = Separator()
        print(json.dumps(separator.list_supported_model_files(), indent=4, sort_keys=True))
        exit(0)

    if not hasattr(args, "audio_file"):
        parser.print_help()
        exit(1)

    logger.info(f"Separator version {package_version} beginning with input file: {args.audio_file}")

    # Deliberately import here to avoid loading slow dependencies when just running --help
    from audio_separator.separator import Separator

    separator = Separator(
        log_formatter=log_formatter,
        log_level=log_level,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        enable_denoise=args.denoise,
        normalization_threshold=args.normalization,
        output_single_stem=args.single_stem,
        invert_using_spec=args.invert_spect,
        sample_rate=args.sample_rate,
        mdx_params={"hop_length": args.mdx_hop_length, "segment_size": args.mdx_segment_size, "overlap": args.mdx_overlap, "batch_size": args.mdx_batch_size},
        vr_params={
            "batch_size": args.vr_batch_size,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_enable_tta,
            "enable_post_process": args.vr_enable_post_process,
            "post_process_threshold": args.vr_post_process_threshold,
            "high_end_process": args.vr_high_end_process,
        },
    )

    separator.load_model(args.model_filename)

    output_files = separator.separate(args.audio_file)

    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")
