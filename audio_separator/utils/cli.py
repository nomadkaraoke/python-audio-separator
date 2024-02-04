#!/usr/bin/env python
import argparse
import logging
import json
from importlib import metadata


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(description="Separate audio file into different stems.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=45))

    parser.add_argument("audio_file", nargs="?", help="The audio file path to separate, in any common format.", default=argparse.SUPPRESS)

    package_version = metadata.distribution("audio-separator").version
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}")

    parser.add_argument("--log_level", default="info", help="Optional: logging level, e.g. info, debug, warning (default: %(default)s). Example: --log_level=debug")

    parser.add_argument("--list_models", action="store_true", help="List all supported models and exit.")

    parser.add_argument(
        "--model_filename", default="2_HP-UVR.pth", help="Optional: model filename to be used for separation (default: %(default)s). Example: --model_filename=UVR_MDXNET_KARA_2.onnx"
    )

    parser.add_argument("--model_file_dir", default="/tmp/audio-separator-models/", help="Optional: model files directory (default: %(default)s). Example: --model_file_dir=/app/models")

    parser.add_argument("--output_dir", default=None, help="Optional: directory to write output files (default: <current dir>). Example: --output_dir=/app/separated")

    parser.add_argument("--output_format", default="FLAC", help="Optional: output format for separated files, any common format (default: %(default)s). Example: --output_format=MP3")

    parser.add_argument(
        "--denoise", type=lambda x: (str(x).lower() == "true"), default=False, help="Optional: enable or disable denoising during separation (default: %(default)s). Example: --denoise=True"
    )

    parser.add_argument(
        "--normalization_threshold", type=float, default=0.9, help="Optional: max peak amplitude to normalize input and output audio to (default: %(default)s). Example: --normalization_threshold=0.7"
    )

    parser.add_argument("--single_stem", default=None, help="Optional: output only single stem, either instrumental or vocals. Example: --single_stem=instrumental")

    parser.add_argument(
        "--invert_spect", type=lambda x: (str(x).lower() == "true"), default=False, help="Optional: invert secondary stem using spectogram (default: %(default)s). Example: --invert_spect=True"
    )

    parser.add_argument("--sample_rate", type=int, default=44100, help="Optional: sample_rate (default: %(default)s). Example: --sample_rate=44100")

    parser.add_argument("--mdx_hop_length", type=int, default=1024, help="Optional: mdx_hop_length (default: %(default)s). Example: --mdx_hop_length=1024")
    parser.add_argument("--mdx_segment_size", type=int, default=256, help="Optional: mdx_segment_size (default: %(default)s). Example: --mdx_segment_size=256")
    parser.add_argument("--mdx_overlap", type=float, default=0.25, help="Optional: mdx_overlap (default: %(default)s). Example: --mdx_overlap=0.25")
    parser.add_argument("--mdx_batch_size", type=int, default=1, help="Optional: mdx_batch_size (default: %(default)s). Example: --mdx_batch_size=4")

    parser.add_argument("--vr_batch_size", type=int, default=16, help="Optional: vr_batch_size (default: %(default)s). Example: --vr_batch_size=4")
    parser.add_argument("--vr_window_size", type=int, default=512, help="Optional: vr_window_size (default: %(default)s). Example: --vr_window_size=256")
    parser.add_argument("--vr_aggression", type=int, default=5, help="Optional: vr_aggression (default: %(default)s). Example: --vr_aggression=2")

    parser.add_argument("--vr_enable_tta", type=lambda x: (str(x).lower() == "true"), default=False, help="Optional: vr_enable_tta (default: %(default)s). Example: --vr_enable_tta=True")
    parser.add_argument(
        "--vr_enable_post_process", type=lambda x: (str(x).lower() == "true"), default=False, help="Optional: vr_enable_post_process (default: %(default)s). Example: --vr_enable_post_process=True"
    )
    parser.add_argument("--vr_post_process_threshold", type=float, default=0.2, help="Optional: vr_post_process_threshold (default: %(default)s). Example: --vr_post_process_threshold=0.1")
    parser.add_argument(
        "--vr_high_end_process", type=lambda x: (str(x).lower() == "true"), default=False, help="Optional: vr_high_end_process (default: %(default)s). Example: --vr_high_end_process=True"
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

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
        normalization_threshold=args.normalization_threshold,
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
