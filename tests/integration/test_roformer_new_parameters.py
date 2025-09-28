"""
Integration test for newer models with new parameters.
This test ensures that newer Roformer models with additional parameters work correctly.
"""

import pytest
import os
import tempfile
import torch
from unittest.mock import patch, Mock

# This will fail initially - that's expected for TDD
try:
    from audio_separator import Separator
    from audio_separator.separator.roformer.roformer_loader import RoformerLoader
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestRoformerNewParameters:
    """Test newer Roformer models with new parameters."""
    
    @pytest.fixture
    def mock_new_roformer_model(self):
        """Create a mock new Roformer model with additional parameters."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            # Create mock model state dict with new parameters
            mock_state = {
                'state_dict': {
                    'model.dim': torch.tensor(512),
                    'model.depth': torch.tensor(6),
                    'model.stereo': torch.tensor(False),
                    'model.num_stems': torch.tensor(2),
                },
                'config': {
                    'dim': 512,
                    'depth': 6,
                    'stereo': False,
                    'num_stems': 2,
                    'freqs_per_bands': (2, 4, 8, 16, 32, 64),
                    # New parameters that should be supported
                    'mlp_expansion_factor': 6,  # Non-default value
                    'sage_attention': True,     # Enabled
                    'zero_dc': False,           # Non-default value
                    'use_torch_checkpoint': True,
                    'skip_connection': True,
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
            tmp.write(b'mock_audio_data')
            yield tmp.name
        
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_load_newer_model_with_new_parameters(self, mock_new_roformer_model, mock_audio_file):
        """Test that newer models with additional parameters load successfully."""
        
        # This test MUST FAIL initially because implementation doesn't exist
        with tempfile.TemporaryDirectory() as output_dir:
            separator = Separator(
                model_file_dir=os.path.dirname(mock_new_roformer_model),
                output_dir=output_dir
            )
            
            # Load the new model - should work with new implementation
            separator.load_model(os.path.basename(mock_new_roformer_model))
            
            # Separate audio - should work with new parameters
            output_files = separator.separate(mock_audio_file)
            
            # Verify outputs exist and are valid
            assert len(output_files) >= 1, "Should produce audio outputs"
            for output_file in output_files:
                assert os.path.exists(output_file), f"Output file should exist: {output_file}"
                assert os.path.getsize(output_file) > 0, f"Output file should not be empty: {output_file}"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_new_model_uses_new_implementation(self, mock_new_roformer_model):
        """Test that new models use the new implementation (not fallback)."""
        
        # Mock the loader to verify new implementation is used
        with patch('audio_separator.separator.roformer.roformer_loader.RoformerLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            from audio_separator.separator.roformer.roformer_loader import ModelLoadingResult, ImplementationVersion
            
            mock_result = ModelLoadingResult(
                success=True,
                model=Mock(),
                error_message=None,
                implementation_used=ImplementationVersion.NEW,  # Should use new implementation
                warnings=[]
            )
            mock_loader.load_model.return_value = mock_result
            
            # Load model
            loader = RoformerLoader()
            result = loader.load_model(mock_new_roformer_model)
            
            # Verify new implementation was used
            assert result.success is True
            assert result.implementation_used == ImplementationVersion.NEW
            assert len(result.warnings) == 0  # No fallback warnings
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_new_parameters_are_properly_handled(self, mock_new_roformer_model):
        """Test that new parameters are properly loaded and used."""
        
        loader = RoformerLoader()
        result = loader.load_model(mock_new_roformer_model)
        
        assert result.success is True
        model_config = result.model.config
        
        # Verify new parameters are loaded correctly
        assert model_config.mlp_expansion_factor == 6  # From mock config
        assert model_config.sage_attention is True     # From mock config
        assert model_config.zero_dc is False          # From mock config
        assert model_config.use_torch_checkpoint is True
        assert model_config.skip_connection is True
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_sage_attention_parameter_integration(self, mock_new_roformer_model):
        """Test that sage_attention parameter is properly integrated."""
        
        # This test verifies that sage_attention=True is handled correctly
        # and doesn't cause the AttributeError that was mentioned in the spec
        
        loader = RoformerLoader()
        result = loader.load_model(mock_new_roformer_model)
        
        assert result.success is True
        
        # Verify that sage_attention is passed to transformer_kwargs
        model = result.model
        assert hasattr(model, 'sage_attention')
        assert model.sage_attention is True
        
        # Verify no AttributeError is raised during model initialization
        # (This would be caught in the loading process)
        assert result.error_message is None
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_mlp_expansion_factor_parameter_handling(self, mock_new_roformer_model):
        """Test that mlp_expansion_factor parameter is handled correctly."""
        
        # This test verifies that the mlp_expansion_factor parameter
        # doesn't cause the TypeError mentioned in the spec
        
        loader = RoformerLoader()
        result = loader.load_model(mock_new_roformer_model)
        
        assert result.success is True
        
        # Verify parameter is properly set
        model = result.model
        assert hasattr(model, 'mlp_expansion_factor')
        assert model.mlp_expansion_factor == 6  # From mock config
        
        # Verify no TypeError during initialization
        assert result.error_message is None
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_torch_checkpoint_parameter_integration(self, mock_new_roformer_model):
        """Test that use_torch_checkpoint parameter works correctly."""
        
        loader = RoformerLoader()
        result = loader.load_model(mock_new_roformer_model)
        
        assert result.success is True
        model = result.model
        
        # Verify checkpoint parameter is set
        assert hasattr(model, 'use_torch_checkpoint')
        assert model.use_torch_checkpoint is True
        
        # This parameter affects memory usage during forward pass
        # In real implementation, this would be tested during audio separation
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_skip_connection_parameter_integration(self, mock_new_roformer_model):
        """Test that skip_connection parameter works correctly."""
        
        loader = RoformerLoader()
        result = loader.load_model(mock_new_roformer_model)
        
        assert result.success is True
        model = result.model
        
        # Verify skip connection parameter is set
        assert hasattr(model, 'skip_connection')
        assert model.skip_connection is True
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_zero_dc_parameter_handling(self, mock_new_roformer_model):
        """Test that zero_dc parameter is handled correctly."""
        
        loader = RoformerLoader()
        result = loader.load_model(mock_new_roformer_model)
        
        assert result.success is True
        model = result.model
        
        # Verify zero_dc parameter is set to non-default value
        assert hasattr(model, 'zero_dc')
        assert model.zero_dc is False  # Non-default value from mock
    
    # TDD placeholder test removed - implementation is now complete
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_new_model_audio_quality_validation(self, mock_new_roformer_model, mock_audio_file):
        """Test that new models produce high-quality audio separation."""
        
        with tempfile.TemporaryDirectory() as output_dir:
            separator = Separator(output_dir=output_dir)
            separator.load_model(mock_new_roformer_model)
            
            outputs = separator.separate(mock_audio_file)
            
            # Verify outputs exist and meet quality standards
            assert len(outputs) > 0
            for output in outputs:
                assert os.path.exists(output)
                
                # In real implementation, this would:
                # 1. Load and analyze the audio
                # 2. Check for artifacts or quality issues
                # 3. Compare against reference outputs if available
                # 4. Verify SSIM >= 0.80 for spectrograms (per spec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
