"""
Unit tests for audio chunking functionality.
Tests the AudioChunker class for splitting and merging audio files.
"""

import pytest
import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from pydub import AudioSegment
from pydub.generators import Sine

from audio_separator.separator.audio_chunking import AudioChunker
from audio_separator.separator.exceptions import AudioExportError


class TestAudioChunker:
    """Test cases for AudioChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunk_duration_seconds = 10.0  # 10 seconds
        self.logger = logging.getLogger(__name__)
        self.chunker = AudioChunker(self.chunk_duration_seconds, self.logger)

    def test_initialization(self):
        """Test AudioChunker initialization."""
        assert self.chunker.chunk_duration_ms == 10000
        assert self.chunker.logger is not None

    def test_initialization_with_custom_logger(self):
        """Test AudioChunker initialization with custom logger."""
        custom_logger = Mock()
        chunker = AudioChunker(5.0, custom_logger)
        assert chunker.chunk_duration_ms == 5000
        assert chunker.logger == custom_logger

    def test_should_chunk_true(self):
        """Test should_chunk returns True for files longer than chunk duration."""
        # File duration 15 seconds, chunk duration 10 seconds
        assert self.chunker.should_chunk(15.0) is True

    def test_should_chunk_false(self):
        """Test should_chunk returns False for files shorter than chunk duration."""
        # File duration 5 seconds, chunk duration 10 seconds
        assert self.chunker.should_chunk(5.0) is False

    def test_should_chunk_exact_boundary(self):
        """Test should_chunk at exact boundary."""
        # File duration exactly 10 seconds, chunk duration 10 seconds
        assert self.chunker.should_chunk(10.0) is False

    def test_should_chunk_just_over_boundary(self):
        """Test should_chunk just over boundary."""
        # File duration 10.1 seconds, chunk duration 10 seconds
        assert self.chunker.should_chunk(10.1) is True

    @patch('audio_separator.separator.audio_chunking.AudioSegment.from_file')
    @patch('audio_separator.separator.audio_chunking.os.path.exists')
    @patch('audio_separator.separator.audio_chunking.os.makedirs')
    def test_split_audio_basic(self, _mock_makedirs, mock_exists, mock_from_file):
        """Test basic audio splitting."""
        # Mock audio file (30 seconds)
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=30000)  # 30 seconds in ms
        mock_audio.__getitem__ = Mock(side_effect=lambda _: mock_audio)
        mock_audio.export = Mock()
        mock_from_file.return_value = mock_audio
        mock_exists.return_value = True

        temp_dir = tempfile.mkdtemp()
        try:
            chunk_paths = self.chunker.split_audio("test.wav", temp_dir)

            # Should create 3 chunks (30s / 10s = 3)
            assert len(chunk_paths) == 3
            assert all("chunk_" in path for path in chunk_paths)
            assert mock_audio.export.call_count == 3

        finally:
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @patch('audio_separator.separator.audio_chunking.AudioSegment.from_file')
    @patch('audio_separator.separator.audio_chunking.os.path.exists')
    @patch('audio_separator.separator.audio_chunking.os.makedirs')
    def test_split_audio_uneven_chunks(self, _mock_makedirs, mock_exists, mock_from_file):
        """Test splitting audio with uneven chunk sizes."""
        # Mock audio file (25 seconds)
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=25000)  # 25 seconds in ms
        mock_audio.__getitem__ = Mock(side_effect=lambda _: mock_audio)
        mock_audio.export = Mock()
        mock_from_file.return_value = mock_audio
        mock_exists.return_value = True

        temp_dir = tempfile.mkdtemp()
        try:
            chunk_paths = self.chunker.split_audio("test.wav", temp_dir)

            # Should create 3 chunks (ceil(25s / 10s) = 3)
            # First two chunks: 10s each, last chunk: 5s
            assert len(chunk_paths) == 3

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_split_audio_file_not_found(self, tmp_path):
        """Test split_audio with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.chunker.split_audio("nonexistent.wav", str(tmp_path))

    @patch('audio_separator.separator.audio_chunking.AudioSegment.from_file')
    @patch('audio_separator.separator.audio_chunking.os.path.exists')
    def test_merge_chunks_basic(self, mock_exists, mock_from_file, tmp_path):
        """Test basic chunk merging."""
        # Mock chunk files
        mock_chunk1 = Mock()
        mock_chunk2 = Mock()
        mock_combined = Mock()
        mock_combined.export = Mock(side_effect=lambda path, **_kwargs: Path(path).write_bytes(b"audio"))

        # Setup mock to return chunks and allow addition
        mock_from_file.side_effect = [mock_chunk1, mock_chunk2]
        mock_exists.return_value = True

        # Mock AudioSegment.empty() and addition
        with patch('audio_separator.separator.audio_chunking.AudioSegment.empty') as mock_empty:
            mock_empty.return_value = mock_combined
            mock_combined.__add__ = Mock(side_effect=[mock_combined, mock_combined])
            mock_combined.__len__ = Mock(return_value=20000)

            chunk_paths = ["chunk1.wav", "chunk2.wav"]
            output_path = self.chunker.merge_chunks(chunk_paths, str(tmp_path / "output.wav"))

            assert output_path == str(tmp_path / "output.wav")
            assert mock_combined.export.called

    def test_merge_chunks_empty_list(self, tmp_path):
        """Test merge_chunks with empty chunk list."""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            self.chunker.merge_chunks([], str(tmp_path / "output.wav"))

    @patch('audio_separator.separator.audio_chunking.os.path.exists')
    def test_merge_chunks_missing_file(self, mock_exists, tmp_path):
        """Test merge_chunks with missing chunk file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Chunk file not found"):
            self.chunker.merge_chunks(["missing.wav"], str(tmp_path / "output.wav"))

    def test_chunk_duration_calculation(self):
        """Test chunk duration calculation."""
        chunker_5s = AudioChunker(5.0, self.logger)
        assert chunker_5s.chunk_duration_ms == 5000

        chunker_60s = AudioChunker(60.0, self.logger)
        assert chunker_60s.chunk_duration_ms == 60000

        chunker_half = AudioChunker(0.5, self.logger)
        assert chunker_half.chunk_duration_ms == 500


class TestAudioChunkerIntegration:
    """Integration tests with actual audio segment creation."""

    def test_split_and_merge_round_trip(self):
        """Test splitting and merging produces valid output."""
        # Create a simple test audio segment (silence)
        audio = AudioSegment.silent(duration=15000)  # 15 seconds

        temp_dir = tempfile.mkdtemp()
        try:
            # Save test audio
            input_path = os.path.join(temp_dir, "test_input.wav")
            audio.export(input_path, format="wav")

            # Split
            chunker = AudioChunker(5.0)  # 5-second chunks
            chunk_dir = os.path.join(temp_dir, "chunks")
            chunk_paths = chunker.split_audio(input_path, chunk_dir)

            # Should create 3 chunks
            assert len(chunk_paths) == 3
            assert all(os.path.exists(path) for path in chunk_paths)

            # Merge
            output_path = os.path.join(temp_dir, "test_output.wav")
            merged_path = chunker.merge_chunks(chunk_paths, output_path)

            # Verify output exists and has similar duration
            assert os.path.exists(merged_path)
            merged_audio = AudioSegment.from_file(merged_path)

            # Duration should be close (within 100ms due to encoding)
            assert abs(len(merged_audio) - len(audio)) < 100

        finally:
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_split_with_different_formats(self):
        """Test splitting works with different audio formats."""
        audio = AudioSegment.silent(duration=10000)  # 10 seconds

        temp_dir = tempfile.mkdtemp()
        try:
            # Test with .wav extension
            input_wav = os.path.join(temp_dir, "test.wav")
            audio.export(input_wav, format="wav")

            chunker = AudioChunker(5.0)
            chunk_dir = os.path.join(temp_dir, "chunks_wav")
            chunk_paths = chunker.split_audio(input_wav, chunk_dir)

            assert len(chunk_paths) == 2
            assert all(path.endswith(".wav") for path in chunk_paths)

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_merge_failure_preserves_existing_target_and_removes_partial_file(self, tmp_path):
        chunk_path = tmp_path / "chunk.wav"
        AudioSegment.silent(duration=10).export(chunk_path, format="wav")
        output_path = tmp_path / "merged.wav"
        output_path.write_bytes(b"original")

        def export_partial_then_fail(path, **_kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("ffmpeg failed")

        with patch.object(AudioSegment, "export", side_effect=export_partial_then_fail):
            with pytest.raises(AudioExportError):
                AudioChunker(5.0).merge_chunks([str(chunk_path)], str(output_path))

        assert output_path.read_bytes() == b"original"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["chunk.wav", "merged.wav"]

    def test_merge_closes_pydub_export_handle_before_replace(self, tmp_path):
        chunk_path = tmp_path / "chunk.wav"
        AudioSegment.silent(duration=10).export(chunk_path, format="wav")
        export_handle = Mock()

        def export_audio(path, **_kwargs):
            Path(path).write_bytes(b"encoded audio")
            return export_handle

        with patch.object(AudioSegment, "export", side_effect=export_audio):
            AudioChunker(5.0).merge_chunks([str(chunk_path)], str(tmp_path / "merged.wav"))

        export_handle.close.assert_called_once_with()

    def test_merge_preserves_duration_of_a_silent_intermediate_chunk(self, tmp_path):
        chunks = [Sine(440).to_audio_segment(duration=100), AudioSegment.silent(duration=150), Sine(440).to_audio_segment(duration=200)]
        chunk_paths = []
        for index, chunk in enumerate(chunks):
            chunk_path = tmp_path / f"chunk-{index}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(str(chunk_path))

        output_path = tmp_path / "merged.wav"
        AudioChunker(5.0).merge_chunks(chunk_paths, str(output_path))

        decoded = AudioSegment.from_file(output_path)
        assert abs(len(decoded) - 450) <= 2
        assert decoded[110:240].rms == 0

class TestAudioChunkerEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_file(self):
        """Test chunking very short file (shorter than chunk duration)."""
        audio = AudioSegment.silent(duration=2000)  # 2 seconds

        temp_dir = tempfile.mkdtemp()
        try:
            input_path = os.path.join(temp_dir, "short.wav")
            audio.export(input_path, format="wav")

            chunker = AudioChunker(10.0)  # 10-second chunks

            # Should still work, creating just 1 chunk
            chunk_dir = os.path.join(temp_dir, "chunks")
            chunk_paths = chunker.split_audio(input_path, chunk_dir)

            assert len(chunk_paths) == 1

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_exact_multiple_of_chunk_size(self):
        """Test file that's exact multiple of chunk size."""
        audio = AudioSegment.silent(duration=20000)  # 20 seconds

        temp_dir = tempfile.mkdtemp()
        try:
            input_path = os.path.join(temp_dir, "exact.wav")
            audio.export(input_path, format="wav")

            chunker = AudioChunker(10.0)  # 10-second chunks

            chunk_dir = os.path.join(temp_dir, "chunks")
            chunk_paths = chunker.split_audio(input_path, chunk_dir)

            # Should create exactly 2 chunks
            assert len(chunk_paths) == 2

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
