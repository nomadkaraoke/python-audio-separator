#!/usr/bin/env python
import argparse
import logging
import pkg_resources
from audio_separator import Separator

LOG_LEVEL = logging.DEBUG


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    log_handler = logging.StreamHandler()

    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    logger.debug("Parsing CLI args")

    parser = argparse.ArgumentParser(description="Separate audio file into different stems.")

    parser.add_argument("audio_file", nargs="?", help="The audio file path to separate.", default=argparse.SUPPRESS)
    parser.add_argument("--model_name", default="UVR_MDXNET_KARA_2", help="Optional: model name to be used for separation.")
    parser.add_argument("--model_file_dir", default="/tmp/audio-separator-models/", help="Optional: model files directory.")

    parser.add_argument("--use_cuda", action="store_true", help="Optional: use Nvidia GPU with CUDA for separation.")

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory where the separated files will be saved. If not specified, outputs to current dir.",
    )
    parser.add_argument("--version", action="store_true", help="Show the version number and exit")

    args = parser.parse_args()

    if args.version:
        version = pkg_resources.get_distribution("audio-separator").version
        print(f"audio-separator version: {version}")
        exit(0)

    if not hasattr(args, "audio_file"):
        parser.print_help()
        exit(1)

    logger.info(f"Separator beginning with input file: {args.audio_file}")

    separator = Separator(
        args.audio_file,
        model_name=args.model_name,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        use_cuda=args.use_cuda,
        log_formatter=log_formatter,
        log_level=LOG_LEVEL,
    )
    primary_stem_path, secondary_stem_path = separator.separate()

    logger.info(f"Separation complete! Output files: {primary_stem_path} {secondary_stem_path}")


if __name__ == "__main__":
    main()
