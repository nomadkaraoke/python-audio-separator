"""
Unit tests for Separator class chunking functionality with multi-stem support.
Tests the _process_with_chunking() method for 2, 4, and 6-stem models.
"""

import pytest
import os
import re
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock, call
from pydub import AudioSegment

from audio_separator.separator.separator import Separator


class TestSeparatorChunking:
    """Test cases for Separator chunking with multi-stem models."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_process_with_chunking_2_stems(self, mock_chunker_class):
        """Test chunking with 2-stem model (Vocals, Instrumental)."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
            os.path.join(self.temp_dir, "chunk_0001.wav"),
        ]

        # Mock _separate_file to return 2 stems per chunk
        separator._separate_file = Mock(side_effect=[
            # Chunk 1 output
            [
                os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav"),
                os.path.join(self.temp_dir, "chunk_0000_(Instrumental).wav"),
            ],
            # Chunk 2 output
            [
                os.path.join(self.temp_dir, "chunk_0001_(Vocals).wav"),
                os.path.join(self.temp_dir, "chunk_0001_(Instrumental).wav"),
            ],
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify merge_chunks was called twice (once per stem)
        assert mock_chunker.merge_chunks.call_count == 2

        # Verify output contains 2 files
        assert len(result) == 2

        # Verify stem names in output
        output_stems = [os.path.basename(path) for path in result]
        assert any("Instrumental" in name for name in output_stems)
        assert any("Vocals" in name for name in output_stems)

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_process_with_chunking_4_stems(self, mock_chunker_class):
        """Test chunking with 4-stem Demucs model (Drums, Bass, Other, Vocals)."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
            os.path.join(self.temp_dir, "chunk_0001.wav"),
        ]

        # Mock _separate_file to return 4 stems per chunk
        separator._separate_file = Mock(side_effect=[
            # Chunk 1 output (4 stems)
            [
                os.path.join(self.temp_dir, "chunk_0000_(Drums).wav"),
                os.path.join(self.temp_dir, "chunk_0000_(Bass).wav"),
                os.path.join(self.temp_dir, "chunk_0000_(Other).wav"),
                os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav"),
            ],
            # Chunk 2 output (4 stems)
            [
                os.path.join(self.temp_dir, "chunk_0001_(Drums).wav"),
                os.path.join(self.temp_dir, "chunk_0001_(Bass).wav"),
                os.path.join(self.temp_dir, "chunk_0001_(Other).wav"),
                os.path.join(self.temp_dir, "chunk_0001_(Vocals).wav"),
            ],
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify merge_chunks was called 4 times (once per stem)
        assert mock_chunker.merge_chunks.call_count == 4

        # Verify output contains 4 files
        assert len(result) == 4

        # Verify all 4 stem names in output
        output_stems = [os.path.basename(path) for path in result]
        assert any("Drums" in name for name in output_stems), f"Drums not found in {output_stems}"
        assert any("Bass" in name for name in output_stems), f"Bass not found in {output_stems}"
        assert any("Other" in name for name in output_stems), f"Other not found in {output_stems}"
        assert any("Vocals" in name for name in output_stems), f"Vocals not found in {output_stems}"

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_process_with_chunking_6_stems(self, mock_chunker_class):
        """Test chunking with 6-stem Demucs model."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        # Mock _separate_file to return 6 stems
        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "chunk_0000_(Bass).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Drums).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Other).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Guitar).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Piano).wav"),
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify merge_chunks was called 6 times (once per stem)
        assert mock_chunker.merge_chunks.call_count == 6

        # Verify output contains 6 files
        assert len(result) == 6

        # Verify all 6 stem names in output
        output_stems = [os.path.basename(path) for path in result]
        assert any("Bass" in name for name in output_stems)
        assert any("Drums" in name for name in output_stems)
        assert any("Other" in name for name in output_stems)
        assert any("Vocals" in name for name in output_stems)
        assert any("Guitar" in name for name in output_stems)
        assert any("Piano" in name for name in output_stems)

    def test_stem_name_extraction_from_filename(self):
        """Test regex extraction of stem names from chunk filenames."""
        import re

        # Test standard patterns
        test_cases = [
            ("chunk_0000_(Vocals).wav", "Vocals"),
            ("chunk_0001_(Instrumental).wav", "Instrumental"),
            ("chunk_0000_(Drums).wav", "Drums"),
            ("chunk_0000_(Bass).wav", "Bass"),
            ("chunk_0000_(Other).wav", "Other"),
            ("chunk_0000_(Guitar).wav", "Guitar"),
            ("chunk_0000_(Piano).wav", "Piano"),
            ("test_audio_(Vocals).flac", "Vocals"),
            ("long_filename_with_spaces_(Backing Vocals).mp3", "Backing Vocals"),
        ]

        pattern = r'_\(([^)]+)\)'

        for filename, expected_stem in test_cases:
            match = re.search(pattern, filename)
            assert match is not None, f"Pattern did not match for {filename}"
            assert match.group(1) == expected_stem, f"Expected {expected_stem}, got {match.group(1)}"

    def test_stem_name_extraction_fallback(self):
        """Test fallback behavior when filename pattern doesn't match."""
        import re

        # Test filenames without the _(StemName) pattern
        test_cases = [
            "chunk_0000.wav",
            "output.mp3",
            "vocals_only.flac",
        ]

        pattern = r'_\(([^)]+)\)'

        for filename in test_cases:
            match = re.search(pattern, filename)
            # Should not match - fallback logic should kick in
            assert match is None, f"Pattern should not match for {filename}"

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_chunking_preserves_stem_order(self, mock_chunker_class):
        """Test that stems are merged in sorted order."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        # Mock _separate_file with intentionally unsorted output
        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Drums).wav"),
            os.path.join(self.temp_dir, "chunk_0000_(Bass).wav"),
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify stems are processed in sorted order (Bass, Drums, Vocals)
        output_stems = [os.path.basename(path) for path in result]

        # Extract stem names from output
        stem_names = []
        pattern = r'_\(([^)]+)\)'
        for name in output_stems:
            match = re.search(pattern, name)
            if match:
                stem_names.append(match.group(1))

        # Verify sorted order
        assert stem_names == sorted(stem_names), f"Stems not in sorted order: {stem_names}"


class TestSeparatorChunkingLogic:
    """Test internal logic and state management of chunking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_state_restoration_after_chunking(self, mock_chunker_class):
        """Test that separator state is restored after chunking."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = "/original/output/dir"
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = "/original/model/output"

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        # Track state changes during _separate_file call
        state_during_processing = {}
        def track_state(chunk_path, custom_names):
            state_during_processing['chunk_duration'] = separator.chunk_duration
            state_during_processing['output_dir'] = separator.output_dir
            state_during_processing['model_output_dir'] = separator.model_instance.output_dir
            return [os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav")]

        separator._separate_file = Mock(side_effect=track_state)

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify state was modified during processing
        assert state_during_processing['chunk_duration'] is None
        assert state_during_processing['output_dir'] != "/original/output/dir"
        assert state_during_processing['model_output_dir'] != "/original/model/output"

        # Verify state was restored after processing
        assert separator.chunk_duration == 10.0
        assert separator.output_dir == "/original/output/dir"
        assert separator.model_instance.output_dir == "/original/model/output"

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_gpu_cache_cleared_between_chunks(self, mock_chunker_class):
        """Test that GPU cache is cleared after each chunk."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir
        separator.model_instance.clear_gpu_cache = Mock()

        # Mock chunker behavior - 3 chunks
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
            os.path.join(self.temp_dir, "chunk_0001.wav"),
            os.path.join(self.temp_dir, "chunk_0002.wav"),
        ]

        # Mock _separate_file
        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "chunk_(Vocals).wav"),
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify clear_gpu_cache was called 3 times (once per chunk)
        assert separator.model_instance.clear_gpu_cache.call_count == 3

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_temp_directory_cleanup(self, mock_chunker_class):
        """Test that temporary directory is cleaned up."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "chunk_(Vocals).wav"),
        ])

        # Track temporary directory creation
        temp_dirs_created = []
        original_mkdtemp = tempfile.mkdtemp
        def track_mkdtemp(prefix=None):
            temp_dir = original_mkdtemp(prefix=prefix)
            temp_dirs_created.append(temp_dir)
            return temp_dir

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        with patch('tempfile.mkdtemp', side_effect=track_mkdtemp):
            RealSeparator._process_with_chunking(
                separator,
                os.path.join(self.temp_dir, "test.wav"),
                custom_output_names=None
            )

        # Verify temporary directories were cleaned up
        for temp_dir in temp_dirs_created:
            if temp_dir.startswith('/') or temp_dir.startswith('C:'):  # Real temp dir
                assert not os.path.exists(temp_dir), f"Temp directory {temp_dir} was not cleaned up"

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_error_handling_with_state_restoration(self, mock_chunker_class):
        """Test that state is restored even when error occurs."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = "/original/output/dir"
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = "/original/model/output"

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        # Mock _separate_file to raise exception
        separator._separate_file = Mock(side_effect=Exception("Processing error"))

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator

        # Should raise exception
        with pytest.raises(Exception, match="Processing error"):
            RealSeparator._process_with_chunking(
                separator,
                os.path.join(self.temp_dir, "test.wav"),
                custom_output_names=None
            )

        # Verify state was restored despite error
        assert separator.chunk_duration == 10.0
        assert separator.output_dir == "/original/output/dir"
        assert separator.model_instance.output_dir == "/original/model/output"

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_audio_chunker_initialization(self, mock_chunker_class):
        """Test that AudioChunker is initialized with correct parameters."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 15.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "chunk_(Vocals).wav"),
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Verify AudioChunker was initialized with correct parameters
        mock_chunker_class.assert_called_once_with(15.0, separator.logger)

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_custom_output_names_parameter(self, mock_chunker_class):
        """Test that custom_output_names parameter is handled correctly."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav"),
        ])

        custom_names = {"Vocals": "my_custom_vocals"}

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=custom_names
        )

        # Verify custom name was used in output
        assert len(result) == 1
        assert "my_custom_vocals" in os.path.basename(result[0])


class TestSeparatorChunkingEdgeCases:
    """Test edge cases for multi-stem chunking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_empty_output_handling(self, mock_chunker_class):
        """Test handling when a chunk produces no output files."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        # Mock _separate_file to return empty list
        separator._separate_file = Mock(return_value=[])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Should return empty list
        assert len(result) == 0
        # merge_chunks should not be called
        assert mock_chunker.merge_chunks.call_count == 0

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_inconsistent_stem_count_across_chunks(self, mock_chunker_class):
        """Test handling when different chunks produce different stem counts."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
            os.path.join(self.temp_dir, "chunk_0001.wav"),
        ]

        # Mock _separate_file - first chunk has 2 stems, second has 1 stem
        separator._separate_file = Mock(side_effect=[
            [
                os.path.join(self.temp_dir, "chunk_0000_(Vocals).wav"),
                os.path.join(self.temp_dir, "chunk_0000_(Instrumental).wav"),
            ],
            [
                os.path.join(self.temp_dir, "chunk_0001_(Vocals).wav"),
                # Missing Instrumental stem
            ],
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Should handle inconsistency gracefully
        # Vocals should have 2 chunks, Instrumental should have 1 chunk
        assert len(result) == 2

    @patch('audio_separator.separator.audio_chunking.AudioChunker')
    def test_filename_without_stem_pattern(self, mock_chunker_class):
        """Test fallback when filename doesn't match expected pattern."""
        # Setup mock separator
        separator = Mock(spec=Separator)
        separator.logger = self.logger
        separator.output_dir = self.temp_dir
        separator.output_format = "WAV"
        separator.chunk_duration = 10.0
        separator.model_instance = Mock()
        separator.model_instance.output_dir = self.temp_dir

        # Mock chunker behavior
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.split_audio.return_value = [
            os.path.join(self.temp_dir, "chunk_0000.wav"),
        ]

        # Mock _separate_file - return file without standard naming pattern
        separator._separate_file = Mock(return_value=[
            os.path.join(self.temp_dir, "output_file_without_pattern.wav"),
        ])

        # Import and call the actual method
        from audio_separator.separator.separator import Separator as RealSeparator
        result = RealSeparator._process_with_chunking(
            separator,
            os.path.join(self.temp_dir, "test.wav"),
            custom_output_names=None
        )

        # Should still produce output using fallback naming
        assert len(result) == 1
        # Should use fallback name like "stem_0"
        assert "stem_0" in os.path.basename(result[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
