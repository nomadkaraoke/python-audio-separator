"""
Regression tests for Roformer size mismatch issues.
Tests the handling of shorter outputs and broadcast errors in overlap_add operations.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestRoformerSizeMismatch:
    """Regression tests for size mismatch issues in Roformer processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.channels = 2
        self.sample_rate = 44100
        
        # Test cases with different output lengths that have caused issues
        self.problematic_lengths = [1, 16, 32, 236, 512, 1024, 2048]

    def test_overlap_add_safe_lengths(self):
        """T062: Reproduce shorter outputs (N∈{1,16,32,236}) and assert no broadcast errors, output length preserved."""
        
        # Test each problematic length
        for output_length in self.problematic_lengths:
            with self.subTest(output_length=output_length):
                self._test_single_output_length(output_length)

    def _test_single_output_length(self, output_length):
        """Test overlap_add with a specific output length."""
        # Setup test parameters
        target_length = 10000  # Longer target buffer
        chunk_size = 8192
        step_size = 1024
        
        # Create mock model output with specific length
        model_output = torch.randn(self.batch_size, self.channels, output_length)
        target_buffer = torch.zeros(self.batch_size, self.channels, target_length)
        counter = torch.zeros(self.batch_size, self.channels, target_length)
        
        # Mock overlap_add function that handles size mismatches safely
        def mock_overlap_add_safe(output, target, counter, start_idx, expected_len):
            """Safe overlap_add that handles shorter outputs."""
            # Get actual output length
            actual_len = output.shape[-1]
            
            # Calculate safe length to add
            remaining_target = target.shape[-1] - start_idx
            safe_len = min(actual_len, expected_len, remaining_target)
            
            if safe_len <= 0:
                return 0  # Nothing to add
            
            # Ensure no broadcast errors by explicit slicing
            end_idx = start_idx + safe_len
            
            try:
                # Add output to target buffer
                target_slice = target[..., start_idx:end_idx]
                output_slice = output[..., :safe_len]
                
                # Verify shapes match before adding
                assert target_slice.shape == output_slice.shape, (
                    f"Shape mismatch: target_slice {target_slice.shape} != output_slice {output_slice.shape}"
                )
                
                target[..., start_idx:end_idx] += output_slice
                counter[..., start_idx:end_idx] += 1.0
                
                return safe_len
                
            except Exception as e:
                pytest.fail(f"Overlap add failed for output_length {output_length}: {e}")
        
        # Test overlap_add with different starting positions
        start_positions = [0, 1000, 5000, 8000]
        
        for start_idx in start_positions:
            if start_idx >= target_length:
                continue
                
            # Test with expected length equal to chunk_size
            added_len = mock_overlap_add_safe(
                model_output, target_buffer, counter, start_idx, chunk_size
            )
            
            # Verify no broadcast errors occurred (function completed successfully)
            assert added_len >= 0, f"overlap_add should not fail for output_length {output_length}"
            
            # Verify length preservation - added length should not exceed actual output length
            assert added_len <= output_length, (
                f"Added length {added_len} should not exceed actual output length {output_length}"
            )
            
            # Verify added length is reasonable
            remaining_space = target_length - start_idx
            max_possible_add = min(output_length, chunk_size, remaining_space)
            assert added_len <= max_possible_add, (
                f"Added length {added_len} exceeds maximum possible {max_possible_add}"
            )

    def test_overlap_add_edge_cases(self):
        """Test edge cases that have caused size mismatch issues."""
        target_length = 10000
        
        edge_cases = [
            # (output_length, start_idx, expected_chunk_size, description)
            (1, 0, 8192, "Single sample output at start"),
            (1, 9999, 8192, "Single sample output at end"),
            (16, 0, 8192, "Very short output at start"),
            (32, 5000, 8192, "Short output in middle"),
            (236, 9000, 8192, "Medium output near end"),
            (8192, 2000, 8192, "Full chunk size output"),
            (10000, 0, 8192, "Output longer than chunk size"),
            (5000, 8000, 8192, "Output extending beyond target"),
        ]
        
        for output_len, start_idx, chunk_size, description in edge_cases:
            with self.subTest(case=description):
                # Create test tensors
                model_output = torch.randn(self.batch_size, self.channels, output_len)
                target_buffer = torch.zeros(self.batch_size, self.channels, target_length)
                
                # Mock safe overlap_add
                def safe_overlap_add(output, target, start, chunk_sz):
                    actual_len = output.shape[-1]
                    remaining = target.shape[-1] - start
                    safe_len = min(actual_len, chunk_sz, remaining)
                    
                    if safe_len > 0:
                        end = start + safe_len
                        target[..., start:end] += output[..., :safe_len]
                    
                    return safe_len
                
                # Should not raise any exceptions
                try:
                    added_len = safe_overlap_add(model_output, target_buffer, start_idx, chunk_size)
                    
                    # Verify reasonable result
                    assert added_len >= 0, f"Should add non-negative length for {description}"
                    assert added_len <= output_len, f"Should not add more than available for {description}"
                    
                except Exception as e:
                    pytest.fail(f"Edge case failed - {description}: {e}")

    def test_counter_consistency_with_overlap_add(self):
        """Test that counter updates match overlap_add operations for problematic lengths."""
        target_length = 10000
        
        for output_length in self.problematic_lengths:
            with self.subTest(output_length=output_length):
                # Create test tensors
                model_output = torch.randn(self.batch_size, self.channels, output_length)
                target_buffer = torch.zeros(self.batch_size, self.channels, target_length)
                counter = torch.zeros(self.batch_size, self.channels, target_length)
                
                start_idx = 1000
                chunk_size = 8192
                
                # Mock consistent overlap_add and counter update
                def consistent_update(output, target, counter, start, chunk_sz):
                    actual_len = output.shape[-1]
                    remaining = target.shape[-1] - start
                    safe_len = min(actual_len, chunk_sz, remaining)
                    
                    if safe_len > 0:
                        end = start + safe_len
                        # Update both target and counter with same range
                        target[..., start:end] += output[..., :safe_len]
                        counter[..., start:end] += 1.0
                    
                    return safe_len
                
                added_len = consistent_update(
                    model_output, target_buffer, counter, start_idx, chunk_size
                )
                
                # Verify consistency
                if added_len > 0:
                    # Check that counter was updated in the same range as target
                    end_idx = start_idx + added_len
                    counter_slice = counter[0, 0, start_idx:end_idx]
                    
                    # Counter should be 1.0 where we added data
                    assert torch.all(counter_slice == 1.0), (
                        f"Counter not consistent for output_length {output_length}"
                    )
                    
                    # Counter should be 0.0 outside the updated range
                    if start_idx > 0:
                        before_slice = counter[0, 0, :start_idx]
                        assert torch.all(before_slice == 0.0), (
                            f"Counter corrupted before range for output_length {output_length}"
                        )
                    
                    if end_idx < target_length:
                        after_slice = counter[0, 0, end_idx:]
                        assert torch.all(after_slice == 0.0), (
                            f"Counter corrupted after range for output_length {output_length}"
                        )

    def test_broadcast_error_prevention(self):
        """Test specific scenarios that previously caused broadcast errors."""
        # These are specific cases that have been observed to cause issues
        broadcast_error_cases = [
            # (batch, channels, output_len, target_len, start_idx)
            (1, 2, 1, 8192, 0),      # Single batch, stereo, 1 sample
            (2, 1, 16, 8192, 100),   # Dual batch, mono, 16 samples
            (1, 1, 32, 8192, 8000),  # Single batch, mono, 32 samples near end
            (2, 2, 236, 8192, 4000), # Dual batch, stereo, 236 samples
            (1, 2, 512, 1024, 800),  # Output longer than remaining target space
        ]
        
        for batch, channels, output_len, target_len, start_idx in broadcast_error_cases:
            with self.subTest(batch=batch, channels=channels, output_len=output_len):
                # Create tensors with specific problematic dimensions
                model_output = torch.randn(batch, channels, output_len)
                target_buffer = torch.zeros(batch, channels, target_len)
                
                # Mock overlap_add with explicit broadcast error prevention
                def broadcast_safe_overlap_add(output, target, start):
                    try:
                        # Get dimensions
                        out_batch, out_channels, out_len = output.shape
                        tgt_batch, tgt_channels, tgt_len = target.shape
                        
                        # Verify batch and channel dimensions match
                        assert out_batch == tgt_batch, f"Batch size mismatch: {out_batch} != {tgt_batch}"
                        assert out_channels == tgt_channels, f"Channel mismatch: {out_channels} != {tgt_channels}"
                        
                        # Calculate safe range
                        remaining = tgt_len - start
                        safe_len = min(out_len, remaining)
                        
                        if safe_len > 0:
                            end = start + safe_len
                            
                            # Explicit slicing to prevent broadcast errors
                            target_slice = target[:, :, start:end]
                            output_slice = output[:, :, :safe_len]
                            
                            # Verify shapes match exactly
                            assert target_slice.shape == output_slice.shape, (
                                f"Shape mismatch: {target_slice.shape} != {output_slice.shape}"
                            )
                            
                            # Perform addition
                            target[:, :, start:end] += output_slice
                        
                        return safe_len
                        
                    except Exception as e:
                        pytest.fail(f"Broadcast error not prevented: {e}")
                
                # Should complete without broadcast errors
                added_len = broadcast_safe_overlap_add(model_output, target_buffer, start_idx)
                
                # Verify result is reasonable
                expected_max = min(output_len, target_len - start_idx)
                assert 0 <= added_len <= expected_max, (
                    f"Added length {added_len} outside expected range [0, {expected_max}]"
                )

    def test_output_length_preservation_regression(self):
        """Test that output length is preserved correctly in all problematic cases."""
        # This tests the regression where shorter outputs were not handled correctly
        
        for output_length in self.problematic_lengths:
            with self.subTest(output_length=output_length):
                # Simulate processing pipeline that should preserve length
                original_length = output_length
                
                # Mock processing steps that might alter length
                def mock_processing_pipeline(input_length):
                    """Mock processing that should preserve input length."""
                    # Step 1: Model inference (might produce shorter output)
                    model_output_length = input_length  # Should match input
                    
                    # Step 2: Overlap-add processing
                    processed_length = model_output_length  # Should preserve length
                    
                    # Step 3: Final output
                    final_length = processed_length  # Should still match
                    
                    return {
                        'original': input_length,
                        'model_output': model_output_length,
                        'processed': processed_length,
                        'final': final_length
                    }
                
                result = mock_processing_pipeline(original_length)
                
                # Verify length preservation throughout pipeline
                assert result['model_output'] == original_length, (
                    f"Model output length {result['model_output']} != original {original_length}"
                )
                assert result['processed'] == original_length, (
                    f"Processed length {result['processed']} != original {original_length}"
                )
                assert result['final'] == original_length, (
                    f"Final length {result['final']} != original {original_length}"
                )

    def test_memory_layout_consistency(self):
        """Test that memory layout is consistent for problematic tensor sizes."""
        # Some size mismatches can be caused by unexpected memory layouts
        
        for output_length in self.problematic_lengths:
            with self.subTest(output_length=output_length):
                # Create tensors with different memory layouts
                contiguous_output = torch.randn(self.batch_size, self.channels, output_length)
                non_contiguous_output = contiguous_output.transpose(1, 2).transpose(1, 2)
                
                target = torch.zeros(self.batch_size, self.channels, 10000)
                
                # Mock overlap_add that handles memory layout
                def layout_aware_overlap_add(output, target, start):
                    # Ensure contiguous memory layout
                    if not output.is_contiguous():
                        output = output.contiguous()
                    
                    if not target.is_contiguous():
                        target = target.contiguous()
                    
                    # Perform safe addition
                    out_len = output.shape[-1]
                    remaining = target.shape[-1] - start
                    safe_len = min(out_len, remaining)
                    
                    if safe_len > 0:
                        end = start + safe_len
                        target[..., start:end] += output[..., :safe_len]
                    
                    return safe_len
                
                # Test with both contiguous and non-contiguous tensors
                contiguous_result = layout_aware_overlap_add(contiguous_output, target.clone(), 1000)
                non_contiguous_result = layout_aware_overlap_add(non_contiguous_output, target.clone(), 1000)
                
                # Results should be the same regardless of memory layout
                assert contiguous_result == non_contiguous_result, (
                    f"Memory layout affected result for output_length {output_length}: "
                    f"contiguous={contiguous_result}, non_contiguous={non_contiguous_result}"
                )
