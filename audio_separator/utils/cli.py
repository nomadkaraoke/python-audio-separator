#!/usr/bin/env python
import argparse
import logging
import pkg_resources
from audio_separator import Separator


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(
        description="Separate audio file into different stems.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("audio_file", nargs="?", help="The audio file path to separate, in any common format.", default=argparse.SUPPRESS)

    package_version = pkg_resources.get_distribution("audio-separator").version
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}")

    parser.add_argument(
        "--log_level", default="info", help="Optional: logging level, e.g. info, debug, warning (default: info). Ex: --log_level=debug"
    )

    parser.add_argument(
        "--model_name",
        default="UVR_MDXNET_KARA_2",
        help="Optional: model name to be used for separation (default: UVR_MDXNET_KARA_2). Ex: --model_name=UVR-MDX-NET-Inst_HQ_3",
    )

    parser.add_argument(
        "--model_file_dir",
        default="/tmp/audio-separator-models/",
        help="Optional: model files directory (default: /tmp/audio-separator-models/). Ex: --model_file_dir=/app/models",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional: directory to write output files (default: <current dir>). Ex: --output_dir=/app/separated",
    )

    parser.add_argument(
        "--use_cuda", action="store_true", help="Optional: use Nvidia GPU with CUDA for separation (default: false). Ex: --use_cuda=true"
    )

    parser.add_argument(
        "--output_format", default="FLAC", help="Optional: output format for separated files, any common format. Default: FLAC"
    )

    parser.add_argument(
        "--denoise",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Optional: enable or disable denoising during separation (default: True). Ex: --denoise=False",
    )

    parser.add_argument(
        "--normalize",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Optional: enable or disable normalization during separation (default: True). Ex: --normalize=False",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    if not hasattr(args, "audio_file"):
        parser.print_help()
        exit(1)

    logger.info(f"Separator beginning with input file: {args.audio_file}")

    separator = Separator(
        args.audio_file,
        log_formatter=log_formatter,
        log_level=log_level,
        model_name=args.model_name,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        use_cuda=args.use_cuda,
        output_format=args.output_format,
        denoise_enabled=args.denoise,
        normalization_enabled=args.normalize,
    )
    primary_stem_path, secondary_stem_path = separator.separate()

    logger.info(f"Separation complete! Output files: {primary_stem_path} {secondary_stem_path}")


if __name__ == "__main__":
    main()
