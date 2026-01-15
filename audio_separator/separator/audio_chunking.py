"""Audio chunking utilities for processing large audio files to prevent OOM errors."""

import os
import logging
from typing import List
from pydub import AudioSegment


class AudioChunker:
    """
    Handles splitting and merging of large audio files.

    This class provides utilities to:
    - Split large audio files into fixed-duration chunks
    - Merge processed chunks back together with simple concatenation
    - Determine if a file should be chunked based on its duration

    Example:
        >>> chunker = AudioChunker(chunk_duration_seconds=600)  # 10-minute chunks
        >>> chunk_paths = chunker.split_audio("long_audio.wav", "/tmp/chunks")
        >>> # Process each chunk...
        >>> output_path = chunker.merge_chunks(processed_chunks, "output.wav")
    """

    def __init__(self, chunk_duration_seconds: float, logger: logging.Logger = None):
        """
        Initialize the AudioChunker.

        Args:
            chunk_duration_seconds: Duration of each chunk in seconds
            logger: Optional logger instance for logging operations
        """
        self.chunk_duration_ms = int(chunk_duration_seconds * 1000)
        self.logger = logger or logging.getLogger(__name__)

    def split_audio(self, input_path: str, output_dir: str) -> List[str]:
        """
        Split audio file into fixed-size chunks.

        Args:
            input_path: Path to the input audio file
            output_dir: Directory where chunk files will be saved

        Returns:
            List of paths to the created chunk files

        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If there's an error reading or writing audio files
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.logger.debug(f"Loading audio file: {input_path}")
        audio = AudioSegment.from_file(input_path)

        total_duration_ms = len(audio)
        chunk_paths = []

        # Calculate number of chunks
        num_chunks = (total_duration_ms + self.chunk_duration_ms - 1) // self.chunk_duration_ms
        self.logger.info(f"Splitting {total_duration_ms / 1000:.1f}s audio into {num_chunks} chunks of {self.chunk_duration_ms / 1000:.1f}s each")

        # Get file extension from input
        _, ext = os.path.splitext(input_path)
        if not ext:
            ext = ".wav"  # Default to WAV if no extension

        # Split into chunks
        for i in range(num_chunks):
            start_ms = i * self.chunk_duration_ms
            end_ms = min(start_ms + self.chunk_duration_ms, total_duration_ms)

            chunk = audio[start_ms:end_ms]
            chunk_filename = f"chunk_{i:04d}{ext}"
            chunk_path = os.path.join(output_dir, chunk_filename)

            self.logger.debug(f"Exporting chunk {i + 1}/{num_chunks}: {start_ms / 1000:.1f}s - {end_ms / 1000:.1f}s to {chunk_path}")
            chunk.export(chunk_path, format=ext.lstrip('.'))
            chunk_paths.append(chunk_path)

        return chunk_paths

    def merge_chunks(self, chunk_paths: List[str], output_path: str) -> str:
        """
        Merge processed chunks with simple concatenation.

        Args:
            chunk_paths: List of paths to chunk files to merge
            output_path: Path where the merged output will be saved

        Returns:
            Path to the merged output file

        Raises:
            ValueError: If chunk_paths is empty
            FileNotFoundError: If any chunk file doesn't exist
            IOError: If there's an error reading or writing audio files
        """
        if not chunk_paths:
            raise ValueError("Cannot merge empty list of chunks")

        # Verify all chunks exist
        for chunk_path in chunk_paths:
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        self.logger.info(f"Merging {len(chunk_paths)} chunks into {output_path}")

        # Start with empty audio segment
        combined = AudioSegment.empty()

        # Concatenate all chunks
        for i, chunk_path in enumerate(chunk_paths):
            self.logger.debug(f"Loading chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
            chunk = AudioSegment.from_file(chunk_path)
            combined += chunk  # Simple concatenation

        # Get output format from file extension
        _, ext = os.path.splitext(output_path)
        output_format = ext.lstrip('.') if ext else 'wav'

        self.logger.info(f"Exporting merged audio ({len(combined) / 1000:.1f}s) to {output_path}")
        combined.export(output_path, format=output_format)

        return output_path

    def should_chunk(self, audio_duration_seconds: float) -> bool:
        """
        Determine if file is large enough to benefit from chunking.

        Args:
            audio_duration_seconds: Duration of the audio file in seconds

        Returns:
            True if the file should be chunked, False otherwise
        """
        return audio_duration_seconds > (self.chunk_duration_ms / 1000)
