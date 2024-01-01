#!/usr/bin/env python
import argparse
import logging
import pkg_resources


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
        "--output_format",
        default="FLAC",
        help="Optional: output format for separated files, any common format (default: %(default)s). Example: --output_format=MP3",
    )

    parser.add_argument(
        "--denoise",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Optional: enable or disable denoising during separation (default: %(default)s). Example: --denoise=True",
    )

    parser.add_argument(
        "--normalize",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Optional: enable or disable normalization during separation (default: %(default)s). Example: --normalize=True",
    )

    parser.add_argument(
        "--single_stem",
        default=None,
        help="Optional: output only single stem, either instrumental or vocals. Example: --single_stem=instrumental",
    )

    parser.add_argument(
        "--invert_spect",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Optional: invert secondary stem using spectogram (default: %(default)s). Example: --invert_spect=True",
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Optional: sample_rate (default: %(default)s). Example: --sample_rate=44100",
    )

    parser.add_argument(
        "--hop_length",
        type=int,
        default=1024,
        help="Optional: hop_length (default: %(default)s). Example: --hop_length=1024",
    )

    parser.add_argument(
        "--segment_size",
        type=int,
        default=256,
        help="Optional: segment_size (default: %(default)s). Example: --segment_size=256",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Optional: overlap (default: %(default)s). Example: --overlap=0.25",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Optional: batch_size (default: %(default)s). Example: --batch_size=1",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    if not hasattr(args, "audio_file"):
        parser.print_help()
        exit(1)

    logger.info(f"Separator beginning with input file: {args.audio_file}")

    # Deliberately import here to avoid loading heave dependencies when just running --help
    from audio_separator.separator import Separator

    separator = Separator(
        args.audio_file,
        log_formatter=log_formatter,
        log_level=log_level,
        model_name=args.model_name,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        denoise_enabled=args.denoise,
        normalization_enabled=args.normalize,
        output_single_stem=args.single_stem,
        invert_using_spec=args.invert_spect,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        segment_size=args.segment_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    output_files = separator.separate()

    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")


if __name__ == "__main__":
    main()
