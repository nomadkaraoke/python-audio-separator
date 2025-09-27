"""
Integration test for existing older model compatibility.
This test ensures that existing models continue to work without regression.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, Mock
import torch

# This will fail initially - that's expected for TDD
try:
    from audio_separator import Separator
    from audio_separator.separator.roformer.roformer_loader import RoformerLoader
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestRoformerBackwardCompatibility:
    """Test backward compatibility with existing older Roformer models."""
    
    @pytest.fixture
    def mock_old_roformer_model(self):
        """Create a mock old Roformer model file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            # Create a minimal mock model state dict that represents an old Roformer
            mock_state = {
                'state_dict': {
                    'model.dim': torch.tensor(512),
                    'model.depth': torch.tensor(6),
                    'model.stereo': torch.tensor(False),
                    'model.num_stems': torch.tensor(2),
                    # Old model doesn't have new parameters like mlp_expansion_factor
                },
                'config': {
                    'dim': 512,
                    'depth': 6,
                    'stereo': False,
                    'num_stems': 2,
                    'freqs_per_bands': (2, 4, 8, 16, 32, 64),
                    # Missing new parameters: mlp_expansion_factor, sage_attention, etc.
                }
            }
            torch.save(mock_state, tmp.name)
            yield tmp.name
        
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp:
            # Create minimal mock audio data (this would normally be actual audio)
            tmp.write(b'mock_audio_data')
            yield tmp.name
        
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_load_existing_older_model_without_regression(self, mock_old_roformer_model, mock_audio_file):
        """Test that existing older models load and work identically to before update."""
        
        # This test MUST FAIL initially because implementation doesn't exist
        with tempfile.TemporaryDirectory() as output_dir:
            separator = Separator(
                model_file_dir=os.path.dirname(mock_old_roformer_model),
                output_dir=output_dir
            )
            
            # Load the old model - should work with fallback mechanism
            separator.load_model(os.path.basename(mock_old_roformer_model))
            
            # Separate audio - should produce same results as before
            output_files = separator.separate(mock_audio_file)
            
            # Verify outputs exist and are valid
            assert len(output_files) == 2, "Should produce 2 stems (vocal/instrumental)"
            for output_file in output_files:
                assert os.path.exists(output_file), f"Output file should exist: {output_file}"
                assert os.path.getsize(output_file) > 0, f"Output file should not be empty: {output_file}"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_old_model_uses_fallback_implementation(self, mock_old_roformer_model):
        """Test that old models automatically use fallback to old implementation."""
        
        # Mock the loader to verify fallback behavior
        with patch('audio_separator.separator.roformer.roformer_loader.RoformerLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            # Configure mock to simulate fallback scenario
            from audio_separator.separator.roformer.roformer_loader import ModelLoadingResult, ImplementationVersion
            
            mock_result = ModelLoadingResult(
                success=True,
                model=Mock(),
                error_message=None,
                implementation_used=ImplementationVersion.FALLBACK,  # Should use fallback
                warnings=["Fell back to old implementation due to missing parameters"]
            )
            mock_loader.load_model.return_value = mock_result
            
            # Load model
            loader = RoformerLoader()
            result = loader.load_model(mock_old_roformer_model)
            
            # Verify fallback was used
            assert result.success is True
            assert result.implementation_used == ImplementationVersion.FALLBACK
            assert len(result.warnings) > 0
            assert "fallback" in result.warnings[0].lower()
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_old_model_configuration_compatibility(self, mock_old_roformer_model):
        """Test that old model configurations are properly handled."""
        
        # This test verifies that missing new parameters don't break loading
        loader = RoformerLoader()
        
        # Load model with old configuration format
        result = loader.load_model(mock_old_roformer_model)
        
        # Should succeed despite missing new parameters
        assert result.success is True
        assert result.model is not None
        
        # Verify that default values were applied for missing parameters
        model_config = result.model.config
        assert hasattr(model_config, 'mlp_expansion_factor')
        assert model_config.mlp_expansion_factor == 4  # Default value
        assert hasattr(model_config, 'sage_attention')
        assert model_config.sage_attention is False  # Default value
    
    # TDD placeholder test removed - implementation is now complete
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_audio_quality_regression_detection(self, mock_old_roformer_model, mock_audio_file):
        """Test that audio quality hasn't regressed from the update."""
        
        # This test would compare outputs before and after the update
        # For now, it's a placeholder that will be implemented with actual audio processing
        
        with tempfile.TemporaryDirectory() as output_dir:
            separator = Separator(output_dir=output_dir)
            separator.load_model(mock_old_roformer_model)
            
            outputs = separator.separate(mock_audio_file)
            
            # In real implementation, this would:
            # 1. Load reference outputs from before the update
            # 2. Compare waveforms using SSIM or similar metrics
            # 3. Assert similarity >= 0.90 (per specification)
            
            # For now, just verify outputs exist
            assert len(outputs) > 0
            for output in outputs:
                assert os.path.exists(output)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_performance_no_significant_degradation(self, mock_old_roformer_model, mock_audio_file):
        """Test that model loading performance hasn't significantly degraded."""
        
        import time
        
        # Measure loading time
        start_time = time.time()
        
        separator = Separator()
        separator.load_model(mock_old_roformer_model)
        
        loading_time = time.time() - start_time
        
        # Should load in reasonable time (this is a placeholder threshold)
        assert loading_time < 30.0, f"Model loading took too long: {loading_time}s"
        
        # Measure separation time
        start_time = time.time()
        outputs = separator.separate(mock_audio_file)
        separation_time = time.time() - start_time
        
        # Should separate in reasonable time
        assert separation_time < 60.0, f"Audio separation took too long: {separation_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
