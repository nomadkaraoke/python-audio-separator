"""
Integration test for fallback mechanism activation.
This test ensures fallback from new to old implementation works correctly.
"""

import pytest
import tempfile
import torch

# This will fail initially - that's expected for TDD
try:
    from audio_separator import Separator
    from audio_separator.separator.roformer.roformer_loader import RoformerLoader, ImplementationVersion
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestRoformerFallbackMechanism:
    """Test fallback mechanism activation."""
    
    @pytest.fixture
    def mock_edge_case_model(self):
        """Create a mock model that triggers fallback."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            # Model that might cause new implementation to fail
            mock_state = {
                'config': {
                    'dim': 512, 'depth': 6, 'num_stems': 2,
                    'legacy_parameter': True,  # Unknown to new implementation
                    'freqs_per_bands': (2, 4, 8, 16, 32, 64)
                }
            }
            torch.save(mock_state, tmp.name)
            yield tmp.name
        import os
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_fallback_mechanism_activation(self, mock_edge_case_model):
        """Test that fallback mechanism activates when new implementation fails."""
        
        loader = RoformerLoader()
        result = loader.load_model(mock_edge_case_model)
        
        # Should succeed using fallback
        assert result.success is True
        assert result.implementation_used == ImplementationVersion.FALLBACK
        assert len(result.warnings) > 0
        assert "fallback" in result.warnings[0].lower()
    
    def test_no_implementation_exists_yet(self):
        """TDD verification: This test ensures no implementation exists yet."""
        with pytest.raises(ImportError):
            from audio_separator.separator.roformer.fallback_loader import FallbackLoader


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
