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
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=45),
    )

    parser.add_argument("audio_file", nargs="?", help="The audio file path to separate, in any common format.", default=argparse.SUPPRESS)

    package_version = pkg_resources.get_distribution("audio-separator").version
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}")

    parser.add_argument(
        "--log_level",
        default="info",
        help="Optional: logging level, e.g. info, debug, warning (default: %(default)s). Example: --log_level=debug",
    )

    parser.add_argument(
        "--model_name",
        default="UVR_MDXNET_KARA_2",
        help="Optional: model name to be used for separation (default: %(default)s). Example: --model_name=UVR-MDX-NET-Inst_HQ_3",
    )

    parser.add_argument(
        "--model_file_dir",
        default="/tmp/audio-separator-models/",
        help="Optional: model files directory (default: %(default)s). Example: --model_file_dir=/app/models",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional: directory to write output files (default: <current dir>). Example: --output_dir=/app/separated",
    )

    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Optional: use Nvidia GPU with CUDA for separation (default: %(default)s). Example: --use_cuda=true",
    )

    parser.add_argument(
        "--use_coreml",
        action="store_true",
        help="Optional: use Apple Silicon GPU with CoreML for separation (default: %(default)s). Example: --use_coreml=true",
    )

    parser.add_argument(
        "--output_format",
        default="FLAC",
        help="Optional: output format for separated files, any common format (default: %(default)s). Example: --output_format=MP3",
    )

    parser.add_argument(
        "--denoise",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Optional: enable or disable denoising during separation (default: %(default)s). Example: --denoise=False",
    )

    parser.add_argument(
        "--normalize",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Optional: enable or disable normalization during separation (default: %(default)s). Example: --normalize=False",
    )

    parser.add_argument(
        "--single_stem",
        default=None,
        help="Optional: output only single stem, either instrumental or vocals. Example: --single_stem=instrumental",
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
        use_coreml=args.use_coreml,
        output_format=args.output_format,
        denoise_enabled=args.denoise,
        normalization_enabled=args.normalize,
        output_single_stem=args.single_stem,
    )
    output_files = separator.separate()

    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")


if __name__ == "__main__":
    main()
