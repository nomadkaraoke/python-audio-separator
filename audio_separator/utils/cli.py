#!/usr/bin/env python
import argparse
import datetime
import pkg_resources
from audio_separator import Separator


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().isoformat()
    print(f"{timestamp} - {message}")


def main():
    parser = argparse.ArgumentParser(description="Separate audio file into different stems.")

    parser.add_argument("audio_file", nargs='?', help="The audio file path to separate.", default=argparse.SUPPRESS)
    parser.add_argument("--model_name", default='UVR_MDXNET_KARA_2', help="Optional model name to be used for separation.")
    parser.add_argument("--model_file_dir", default='/tmp/audio-separator-models/', help="Optional model files directory.")
    parser.add_argument("--output_dir", default=None, help="Optional directory where the separated files will be saved. If not specified, outputs to current dir.")
    parser.add_argument('--version', action='store_true', help='Show the version number and exit')

    args = parser.parse_args()

    if args.version:
        version = pkg_resources.get_distribution("audio-separator").version
        print(f"audio-separator version: {version}")
        exit(0)

    if not hasattr(args, 'audio_file'):
        parser.print_help()
        exit(1)

    print_with_timestamp(f'Separator beginning with input file: {args.audio_file}')

    separator = Separator(args.audio_file, model_name=args.model_name, model_file_dir=args.model_file_dir, output_dir=args.output_dir)
    primary_stem_path, secondary_stem_path = separator.separate()

    print_with_timestamp(f'Separation complete! Output files: {primary_stem_path} {secondary_stem_path}')


if __name__ == '__main__':
    main()
