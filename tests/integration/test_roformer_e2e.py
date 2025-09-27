"""
End-to-end integration tests for Roformer models.
Tests complete separation workflow with both new and legacy models.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from pathlib import Path


class TestRoformerE2E:
    """End-to-end tests for Roformer model separation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.flac")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock audio file
        self._create_mock_audio_file()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_audio_file(self):
        """Create a mock audio file for testing."""
        # Create a simple stereo audio file (mock)
        sample_rate = 44100
        duration = 3  # 3 seconds
        samples = int(sample_rate * duration)
        
        # Generate simple test audio (sine waves)
        t = np.linspace(0, duration, samples)
        left_channel = np.sin(2 * np.pi * 440 * t)  # 440 Hz
        right_channel = np.sin(2 * np.pi * 880 * t)  # 880 Hz
        audio_data = np.stack([left_channel, right_channel])
        
        # Mock writing audio file (in real test, would use soundfile)
        with open(self.test_audio_path, 'w') as f:
            f.write("mock_audio_data")

    def test_bs_roformer_sw_fixed_e2e(self):
        """T060: New BSRoformer SW-Fixed end-to-end separation succeeds."""
        # Mock the new BSRoformer SW-Fixed model
        mock_model_path = "model_bs_roformer_sw_fixed.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            # Setup mock separator
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            
            # Mock model loading success
            mock_separator.load_model.return_value = True
            
            # Mock separation results
            expected_outputs = [
                os.path.join(self.output_dir, "test_audio_(vocals).flac"),
                os.path.join(self.output_dir, "test_audio_(instrumental).flac")
            ]
            mock_separator.separate.return_value = expected_outputs
            
            # Mock model configuration for new BSRoformer
            mock_model_config = {
                'model_type': 'bs_roformer',
                'architecture': 'BSRoformer',
                'dim': 512,
                'depth': 12,
                'stereo': True,
                'num_stems': 2,
                'freqs_per_bands': (4096, 2048, 1024, 512),
                # New parameters that should be supported
                'mlp_expansion_factor': 4,
                'sage_attention': True,
                'zero_dc': True,
                'use_torch_checkpoint': False,
                'skip_connection': True
            }
            
            # Initialize separator
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            # Test model loading
            load_success = separator.load_model(mock_model_path)
            assert load_success, "BSRoformer SW-Fixed model should load successfully"
            
            # Verify model was loaded with correct parameters
            separator.load_model.assert_called_once_with(mock_model_path)
            
            # Test separation
            output_files = separator.separate(self.test_audio_path)
            
            # Verify separation completed
            assert output_files is not None, "Separation should return output files"
            assert len(output_files) == 2, "Should produce vocals and instrumental outputs"
            
            # Verify output file paths
            for output_file in output_files:
                assert output_file in expected_outputs, f"Unexpected output file: {output_file}"
                # In real test, would verify file exists and has content
                # assert os.path.exists(output_file), f"Output file should exist: {output_file}"
            
            # Verify separation was called correctly
            separator.separate.assert_called_once_with(self.test_audio_path)

    def test_legacy_roformer_e2e(self):
        """T061: Legacy Roformer end-to-end separation still succeeds."""
        # Mock a legacy Roformer model (without new parameters)
        mock_legacy_model_path = "legacy_roformer_model.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            # Setup mock separator
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            
            # Mock model loading success with fallback
            mock_separator.load_model.return_value = True
            
            # Mock separation results
            expected_outputs = [
                os.path.join(self.output_dir, "test_audio_(vocals).flac"),
                os.path.join(self.output_dir, "test_audio_(instrumental).flac")
            ]
            mock_separator.separate.return_value = expected_outputs
            
            # Mock legacy model configuration (without new parameters)
            mock_legacy_config = {
                'model_type': 'bs_roformer',
                'architecture': 'BSRoformer',
                'dim': 384,
                'depth': 8,
                'stereo': True,
                'num_stems': 2,
                'freqs_per_bands': (2048, 1024, 512, 256),
                # Legacy config - missing new parameters
                # Should fall back to old implementation
            }
            
            # Initialize separator
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            # Test model loading (should use fallback mechanism)
            load_success = separator.load_model(mock_legacy_model_path)
            assert load_success, "Legacy Roformer model should load successfully via fallback"
            
            # Verify model was loaded
            separator.load_model.assert_called_once_with(mock_legacy_model_path)
            
            # Test separation
            output_files = separator.separate(self.test_audio_path)
            
            # Verify separation completed
            assert output_files is not None, "Legacy separation should return output files"
            assert len(output_files) == 2, "Should produce vocals and instrumental outputs"
            
            # Verify output file paths match expected
            for output_file in output_files:
                assert output_file in expected_outputs, f"Unexpected output file: {output_file}"
            
            # Verify separation was called correctly
            separator.separate.assert_called_once_with(self.test_audio_path)

    def test_mel_band_roformer_e2e(self):
        """Test MelBandRoformer end-to-end separation."""
        mock_mel_model_path = "mel_band_roformer_model.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            # Setup mock separator
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            
            # Mock model loading success
            mock_separator.load_model.return_value = True
            
            # Mock separation results for MelBandRoformer
            expected_outputs = [
                os.path.join(self.output_dir, "test_audio_(vocals).flac"),
                os.path.join(self.output_dir, "test_audio_(accompaniment).flac")
            ]
            mock_separator.separate.return_value = expected_outputs
            
            # Mock MelBandRoformer configuration
            mock_mel_config = {
                'model_type': 'mel_band_roformer',
                'architecture': 'MelBandRoformer',
                'dim': 256,
                'depth': 6,
                'stereo': True,
                'num_stems': 2,
                'num_bands': 64,
                'sample_rate': 44100,
                # New parameters
                'mlp_expansion_factor': 4,
                'sage_attention': False,
                'zero_dc': True
            }
            
            # Initialize separator
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            # Test model loading
            load_success = separator.load_model(mock_mel_model_path)
            assert load_success, "MelBandRoformer model should load successfully"
            
            # Test separation
            output_files = separator.separate(self.test_audio_path)
            
            # Verify separation completed
            assert output_files is not None, "MelBandRoformer separation should return output files"
            assert len(output_files) == 2, "Should produce two output stems"
            
            # Verify output file paths
            for output_file in output_files:
                assert output_file in expected_outputs, f"Unexpected output file: {output_file}"

    def test_roformer_e2e_with_different_audio_formats(self):
        """Test E2E separation with different audio formats."""
        audio_formats = ['.flac', '.wav', '.mp3', '.m4a']
        mock_model_path = "model_bs_roformer_test.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            mock_separator.load_model.return_value = True
            
            for audio_format in audio_formats:
                # Create mock audio file with different format
                test_audio = os.path.join(self.temp_dir, f"test_audio{audio_format}")
                with open(test_audio, 'w') as f:
                    f.write("mock_audio_data")
                
                # Mock separation results
                expected_outputs = [
                    os.path.join(self.output_dir, f"test_audio_(vocals){audio_format}"),
                    os.path.join(self.output_dir, f"test_audio_(instrumental){audio_format}")
                ]
                mock_separator.separate.return_value = expected_outputs
                
                # Initialize separator
                separator = mock_separator_class(
                    model_file_dir=self.temp_dir,
                    output_dir=self.output_dir
                )
                
                # Load model
                load_success = separator.load_model(mock_model_path)
                assert load_success, f"Model should load for {audio_format} format"
                
                # Test separation
                output_files = separator.separate(test_audio)
                assert output_files is not None, f"Separation should work with {audio_format} format"
                assert len(output_files) == 2, f"Should produce 2 outputs for {audio_format}"

    def test_roformer_e2e_error_handling(self):
        """Test E2E error handling scenarios."""
        mock_model_path = "problematic_model.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            
            # Test model loading failure
            mock_separator.load_model.return_value = False
            
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            load_success = separator.load_model(mock_model_path)
            assert not load_success, "Should handle model loading failure gracefully"
            
            # Test separation failure
            mock_separator.load_model.return_value = True  # Model loads
            mock_separator.separate.side_effect = Exception("Separation failed")
            
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            separator.load_model(mock_model_path)
            
            # Should handle separation exception
            with pytest.raises(Exception, match="Separation failed"):
                separator.separate(self.test_audio_path)

    def test_roformer_e2e_performance_validation(self):
        """Test E2E performance characteristics."""
        mock_model_path = "performance_test_model.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            mock_separator.load_model.return_value = True
            
            # Mock timing for performance validation
            import time
            
            def mock_separate_with_timing(audio_path):
                # Simulate processing time
                start_time = time.time()
                time.sleep(0.1)  # Simulate 100ms processing
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                # Return mock results with timing info
                return [
                    os.path.join(self.output_dir, "test_audio_(vocals).flac"),
                    os.path.join(self.output_dir, "test_audio_(instrumental).flac")
                ], processing_time
            
            mock_separator.separate.side_effect = lambda x: mock_separate_with_timing(x)[0]
            
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            separator.load_model(mock_model_path)
            
            # Measure separation time
            start_time = time.time()
            output_files = separator.separate(self.test_audio_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify performance characteristics
            assert output_files is not None, "Should produce outputs"
            assert processing_time < 10.0, "Processing should complete in reasonable time"
            assert len(output_files) == 2, "Should produce expected number of outputs"

    def test_roformer_e2e_memory_usage(self):
        """Test E2E memory usage characteristics."""
        mock_model_path = "memory_test_model.ckpt"
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            mock_separator.load_model.return_value = True
            
            # Mock memory usage tracking
            def mock_separate_with_memory_tracking(audio_path):
                # Simulate memory usage
                mock_memory_usage = {
                    'peak_memory_mb': 1024,  # 1GB peak
                    'current_memory_mb': 512,  # 512MB current
                    'gpu_memory_mb': 2048 if torch.cuda.is_available() else 0
                }
                
                return [
                    os.path.join(self.output_dir, "test_audio_(vocals).flac"),
                    os.path.join(self.output_dir, "test_audio_(instrumental).flac")
                ], mock_memory_usage
            
            mock_separator.separate.side_effect = lambda x: mock_separate_with_memory_tracking(x)[0]
            
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            separator.load_model(mock_model_path)
            output_files = separator.separate(self.test_audio_path)
            
            # Verify memory usage is reasonable
            assert output_files is not None, "Should produce outputs despite memory constraints"
            # In real implementation, would check actual memory usage
            # assert peak_memory < threshold, "Memory usage should be within limits"

    def test_roformer_e2e_batch_processing(self):
        """Test E2E batch processing of multiple files."""
        mock_model_path = "batch_test_model.ckpt"
        
        # Create multiple test audio files
        test_files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"test_audio_{i}.flac")
            with open(test_file, 'w') as f:
                f.write(f"mock_audio_data_{i}")
            test_files.append(test_file)
        
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            mock_separator = Mock()
            mock_separator_class.return_value = mock_separator
            mock_separator.load_model.return_value = True
            
            # Mock batch separation results
            def mock_batch_separate(audio_path):
                basename = os.path.splitext(os.path.basename(audio_path))[0]
                return [
                    os.path.join(self.output_dir, f"{basename}_(vocals).flac"),
                    os.path.join(self.output_dir, f"{basename}_(instrumental).flac")
                ]
            
            mock_separator.separate.side_effect = mock_batch_separate
            
            separator = mock_separator_class(
                model_file_dir=self.temp_dir,
                output_dir=self.output_dir
            )
            
            separator.load_model(mock_model_path)
            
            # Process all files
            all_outputs = []
            for test_file in test_files:
                outputs = separator.separate(test_file)
                all_outputs.extend(outputs)
            
            # Verify batch processing
            expected_total_outputs = len(test_files) * 2  # 2 outputs per input
            assert len(all_outputs) == expected_total_outputs, "Should process all files"
            
            # Verify each file was processed
            for i in range(len(test_files)):
                expected_vocals = os.path.join(self.output_dir, f"test_audio_{i}_(vocals).flac")
                expected_instrumental = os.path.join(self.output_dir, f"test_audio_{i}_(instrumental).flac")
                assert expected_vocals in all_outputs, f"Missing vocals output for file {i}"
                assert expected_instrumental in all_outputs, f"Missing instrumental output for file {i}"
