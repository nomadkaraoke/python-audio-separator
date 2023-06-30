#!/usr/bin/env python
import argparse
import datetime
from audio_separator.audio_separator import Separator


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().isoformat()
    print(f"{timestamp} - {message}")


def main():
    parser = argparse.ArgumentParser(description="Separate audio file into different stems.")
    parser.add_argument("audio_file", help="The WAV audio file path to separate.")
    parser.add_argument("--model_name", default='UVR_MDXNET_KARA_2', help="Optional model name to be used for separation.")
    parser.add_argument("--model_file_dir", default='/tmp/audio-separator-models/', help="Optional model files directory.")

    args = parser.parse_args()

    print_with_timestamp(f'Separator beginning with input file: {args.audio_file}')

    separator = Separator(args.audio_file, model_name=args.model_name, model_file_dir=args.model_file_dir)
    primary_stem_path, secondary_stem_path = separator.separate()

    print_with_timestamp(f'Separation complete! Output files: {primary_stem_path} {secondary_stem_path}')


if __name__ == '__main__':
    main()
