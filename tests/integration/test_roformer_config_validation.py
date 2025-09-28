"""
Integration test for configuration validation error handling.
This test ensures invalid configurations are caught with helpful error messages.
"""

import pytest
import tempfile
import torch

# This will fail initially - that's expected for TDD
try:
    from audio_separator import Separator
    from audio_separator.separator.roformer.roformer_loader import ParameterValidationError
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestRoformerConfigValidation:
    """Test configuration validation and error handling."""
    
    @pytest.fixture
    def mock_invalid_config_model(self):
        """Create a mock model with invalid configuration."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            mock_state = {
                'config': {
                    'dim': "invalid_string",  # Should be int
                    'depth': -1,              # Should be positive
                    'attn_dropout': 1.5,      # Should be 0.0-1.0
                    # Missing required 'num_stems'
                }
            }
            torch.save(mock_state, tmp.name)
            yield tmp.name
        import os
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_parameter_validation_error_handling(self, mock_invalid_config_model):
        """Test that invalid configurations raise ParameterValidationError."""
        
        separator = Separator()
        
        with pytest.raises(ParameterValidationError) as exc_info:
            separator.load_model(mock_invalid_config_model)
        
        error = exc_info.value
        assert error.parameter_name is not None
        assert error.expected_type is not None
        assert error.suggested_fix is not None
        assert "integer" in error.suggested_fix or "positive" in error.suggested_fix
    
    # TDD placeholder test removed - implementation is now complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
