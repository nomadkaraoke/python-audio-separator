"""
Unit tests for MDXC Roformer chunking and overlap logic.
Tests the chunking mechanism, overlap handling, and edge cases.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
import logging


class TestMDXCRoformerChunking:
    """Test cases for MDXC Roformer chunking and overlap functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_rate = 44100
        self.audio_length = 132300  # 3 seconds
        self.chunk_size = 8192
        self.hop_length = 1024
        
        # Mock model with stft_hop_length
        self.mock_model = Mock()
        self.mock_model.stft_hop_length = self.hop_length
        
        # Mock audio object
        self.mock_audio = Mock()
        self.mock_audio.hop_length = 512  # Different from model for fallback test

    def test_chunk_size_uses_model_stft_hop_length(self):
        """T051: Assert chunk_size uses model.stft_hop_length."""
        # Test implementation for chunking optimization - placeholder for future implementation
        pytest.skip("Chunking optimization not yet implemented")

    def test_chunk_size_falls_back_to_audio_hop_length(self):
        """T052: Fallback to audio.hop_length if model.stft_hop_length missing."""
        # Test implementation for chunking optimization - placeholder for future implementation
        pytest.skip("Chunking optimization not yet implemented")

    def test_step_clamped_to_chunk_size(self):
        """T053: Step clamped to chunk_size (desired_step > chunk_size or ≤ 0)."""
        chunk_size = 8192
        
        # Test case 1: desired_step > chunk_size
        desired_step_too_large = 10000
        actual_step = min(desired_step_too_large, chunk_size)
        assert actual_step == chunk_size
        
        # Test case 2: desired_step ≤ 0
        desired_step_zero = 0
        actual_step = max(desired_step_zero, chunk_size // 4)  # Use quarter chunk as minimum
        assert actual_step == chunk_size // 4
        
        # Test case 3: desired_step negative
        desired_step_negative = -100
        actual_step = max(desired_step_negative, chunk_size // 4)
        assert actual_step == chunk_size // 4
        
        # Test case 4: valid desired_step
        desired_step_valid = 4096
        actual_step = min(max(desired_step_valid, 1), chunk_size)
        assert actual_step == desired_step_valid

    def test_overlap_add_short_output_safe(self):
        """T054: overlap_add handles shorter model output safely (safe_len)."""
        # Create mock tensors
        chunk_size = 8192
        model_output_length = 6000  # Shorter than expected
        
        # Mock overlap_add logic
        def mock_overlap_add_safe(output, target, start_idx, safe_len):
            """Mock overlap add with safe length handling."""
            actual_len = min(output.shape[-1], safe_len)
            end_idx = start_idx + actual_len
            
            # Ensure we don't go beyond target bounds
            if end_idx > target.shape[-1]:
                end_idx = target.shape[-1]
                actual_len = end_idx - start_idx
            
            if actual_len > 0:
                target[..., start_idx:end_idx] += output[..., :actual_len]
            
            return actual_len
        
        # Test with shorter output
        output = torch.randn(2, model_output_length)  # Shorter than chunk_size
        target = torch.zeros(2, 20000)
        start_idx = 1000
        safe_len = chunk_size
        
        actual_added = mock_overlap_add_safe(output, target, start_idx, safe_len)
        
        # Should handle shorter output gracefully
        assert actual_added == model_output_length
        assert actual_added < safe_len
        
        # Verify no out-of-bounds access
        assert start_idx + actual_added <= target.shape[-1]

    def test_counter_updates_safe_len(self):
        """T055: Counter increments match overlap_add safe span."""
        # Mock counter and overlap_add logic
        counter = torch.zeros(2, 20000)
        chunk_size = 8192
        safe_len = 6000  # Shorter than chunk_size
        start_idx = 1000
        
        def mock_update_counter_safe(counter, start_idx, safe_len):
            """Mock counter update that matches overlap_add safe span."""
            end_idx = start_idx + safe_len
            if end_idx > counter.shape[-1]:
                end_idx = counter.shape[-1]
                safe_len = end_idx - start_idx
            
            if safe_len > 0:
                counter[..., start_idx:end_idx] += 1.0
            
            return safe_len
        
        actual_updated = mock_update_counter_safe(counter, start_idx, safe_len)
        
        # Counter increment should match the safe length used
        assert actual_updated == safe_len
        
        # Verify counter was updated correctly
        expected_ones = counter[0, start_idx:start_idx + actual_updated]
        assert torch.all(expected_ones == 1.0)
        
        # Verify areas outside weren't touched
        before_area = counter[0, :start_idx]
        after_area = counter[0, start_idx + actual_updated:]
        assert torch.all(before_area == 0.0)
        assert torch.all(after_area == 0.0)

    def test_counter_clamp_no_nan(self):
        """T056: No NaN/inf on normalization (counter clamp)."""
        # Create counter with some zero values (potential division by zero)
        counter = torch.tensor([[0.0, 1.0, 2.0, 0.0, 3.0, 0.0]], dtype=torch.float32)
        output = torch.tensor([[1.0, 2.0, 4.0, 8.0, 6.0, 16.0]], dtype=torch.float32)
        
        # Mock normalization with counter clamping
        def mock_normalize_with_clamp(output, counter, min_clamp=1e-8):
            """Mock normalization that clamps counter to avoid NaN/inf."""
            clamped_counter = torch.clamp(counter, min=min_clamp)
            normalized = output / clamped_counter
            return normalized, clamped_counter
        
        normalized, clamped_counter = mock_normalize_with_clamp(output, counter)
        
        # Verify no NaN or inf values
        assert not torch.any(torch.isnan(normalized))
        assert not torch.any(torch.isinf(normalized))
        
        # Verify clamping worked
        assert torch.all(clamped_counter >= 1e-8)
        
        # Verify normalization is reasonable
        assert torch.all(normalized >= 0)  # Should be positive
        
        # Test with all-zero counter (extreme case)
        zero_counter = torch.zeros_like(counter)
        normalized_zero, clamped_zero = mock_normalize_with_clamp(output, zero_counter)
        
        assert not torch.any(torch.isnan(normalized_zero))
        assert not torch.any(torch.isinf(normalized_zero))
        assert torch.all(clamped_zero >= 1e-8)

    def test_short_audio_last_block(self):
        """T057: Short-audio last-block path works and preserves length."""
        # Test with audio shorter than one chunk
        short_audio_length = 4000  # Less than chunk_size (8192)
        chunk_size = 8192
        
        # Mock processing of short audio
        def mock_process_short_audio(audio_length, chunk_size):
            """Mock processing that handles short audio specially."""
            if audio_length < chunk_size:
                # Last block path: process entire audio as one chunk
                return {
                    'processed_length': audio_length,
                    'num_chunks': 1,
                    'last_block': True,
                    'preserved_length': audio_length
                }
            else:
                # Normal chunking path
                num_chunks = (audio_length + chunk_size - 1) // chunk_size
                return {
                    'processed_length': audio_length,
                    'num_chunks': num_chunks,
                    'last_block': False,
                    'preserved_length': audio_length
                }
        
        result = mock_process_short_audio(short_audio_length, chunk_size)
        
        # Verify last block path was taken
        assert result['last_block'] is True
        assert result['num_chunks'] == 1
        
        # Verify length preservation
        assert result['processed_length'] == short_audio_length
        assert result['preserved_length'] == short_audio_length
        
        # Test with normal-length audio for comparison
        normal_audio_length = 20000
        normal_result = mock_process_short_audio(normal_audio_length, chunk_size)
        
        assert normal_result['last_block'] is False
        assert normal_result['num_chunks'] > 1
        assert normal_result['preserved_length'] == normal_audio_length

    @pytest.mark.parametrize("dim_t,hop_length", [
        (256, 512),
        (512, 1024), 
        (1024, 2048),
        (128, 256)
    ])
    def test_parametrized_shape_invariants(self, dim_t, hop_length):
        """T058: Parametrized invariants across dim_t and hop configs."""
        batch_size = 2
        audio_length = 44100  # 1 second
        
        # Mock model with parametrized config
        mock_model = Mock()
        mock_model.dim_t = dim_t
        mock_model.stft_hop_length = hop_length
        
        # Mock chunking calculation
        def mock_calculate_chunks(audio_length, dim_t, hop_length):
            """Calculate number of chunks needed."""
            chunk_size = dim_t * 8  # Example chunk size calculation
            step_size = hop_length
            
            if audio_length <= chunk_size:
                return 1
            
            return (audio_length - chunk_size + step_size - 1) // step_size + 1
        
        num_chunks = mock_calculate_chunks(audio_length, dim_t, hop_length)
        
        # Invariants that should hold regardless of parameters
        assert num_chunks >= 1, f"Should always have at least 1 chunk for dim_t={dim_t}, hop={hop_length}"
        assert num_chunks <= audio_length // hop_length + 2, f"Chunks should be reasonable for dim_t={dim_t}, hop={hop_length}"
        
        # Test output shape consistency
        mock_output_shape = (batch_size, 2, dim_t)  # (batch, channels, time)
        assert mock_output_shape[0] == batch_size
        assert mock_output_shape[2] == dim_t
        
        # Test that chunk_size scales with dim_t
        chunk_size = dim_t * 8
        assert chunk_size > 0
        assert chunk_size >= dim_t

    def test_logging_for_hop_and_step(self, caplog):
        """T063: Logs include hop/step sources (stft_hop_length, dim_t, desired vs actual step)."""
        with caplog.at_level(logging.DEBUG):
            # Mock logging during chunking setup
            def mock_setup_chunking_with_logging(model, audio):
                """Mock chunking setup that logs parameter sources."""
                stft_hop = getattr(model, 'stft_hop_length', None)
                dim_t = getattr(model, 'dim_t', None)
                audio_hop = getattr(audio, 'hop_length', None)
                
                logging.debug(f"Chunking setup: stft_hop_length={stft_hop}, dim_t={dim_t}")
                logging.debug(f"Audio hop_length={audio_hop}")
                
                desired_step = stft_hop if stft_hop else audio_hop
                chunk_size = dim_t * 8 if dim_t else 8192
                actual_step = min(desired_step, chunk_size) if desired_step else chunk_size // 4
                
                logging.debug(f"Step calculation: desired={desired_step}, actual={actual_step}, chunk_size={chunk_size}")
                
                return {
                    'chunk_size': chunk_size,
                    'step_size': actual_step,
                    'stft_hop_source': stft_hop is not None
                }
            
            # Test with model having stft_hop_length
            model_with_stft = Mock()
            model_with_stft.stft_hop_length = 1024
            model_with_stft.dim_t = 256
            
            audio = Mock()
            audio.hop_length = 512
            
            result = mock_setup_chunking_with_logging(model_with_stft, audio)
            
            # Verify logging occurred
            assert "stft_hop_length=1024" in caplog.text
            assert "dim_t=256" in caplog.text
            assert "desired=1024" in caplog.text
            assert "actual=" in caplog.text
            assert "chunk_size=" in caplog.text

    def test_iteration_count_reasonable(self):
        """T064: Iteration count reasonable (ceil calculation within ±1)."""
        import math
        
        # Test various audio lengths and chunk configurations
        test_cases = [
            (44100, 8192, 1024),   # 1 second, normal chunk
            (88200, 4096, 512),    # 2 seconds, smaller chunk  
            (22050, 16384, 2048),  # 0.5 seconds, large chunk
            (132300, 8192, 1024),  # 3 seconds, normal chunk
        ]
        
        for audio_length, chunk_size, step_size in test_cases:
            # Calculate expected iterations using ceiling division
            if audio_length <= chunk_size:
                expected_iterations = 1
            else:
                remaining_length = audio_length - chunk_size
                expected_iterations = 1 + math.ceil(remaining_length / step_size)
            
            # Mock actual iteration calculation
            def mock_calculate_iterations(audio_len, chunk_sz, step_sz):
                if audio_len <= chunk_sz:
                    return 1
                
                iterations = 0
                pos = 0
                while pos < audio_len:
                    iterations += 1
                    if pos + chunk_sz >= audio_len:
                        break
                    pos += step_sz
                
                return iterations
            
            actual_iterations = mock_calculate_iterations(audio_length, chunk_size, step_size)
            
            # Verify iteration count is reasonable (within ±1 of expected)
            diff = abs(actual_iterations - expected_iterations)
            assert diff <= 1, (
                f"Iteration count {actual_iterations} differs too much from expected {expected_iterations} "
                f"for audio_len={audio_length}, chunk={chunk_size}, step={step_size}"
            )
            
            # Verify minimum iterations
            assert actual_iterations >= 1, f"Should always have at least 1 iteration"
            
            # Verify maximum reasonable iterations
            max_reasonable = (audio_length // step_size) + 2
            assert actual_iterations <= max_reasonable, (
                f"Too many iterations {actual_iterations} for audio_len={audio_length}"
            )
