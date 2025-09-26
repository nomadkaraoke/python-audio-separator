"""
Integration test for model type switching (BSRoformer ↔ MelBandRoformer).
This test ensures seamless switching between different Roformer model types.
"""

import pytest
import os
import tempfile
import torch

# This will fail initially - that's expected for TDD
try:
    from audio_separator import Separator
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestRoformerModelSwitching:
    """Test switching between different Roformer model types."""
    
    @pytest.fixture
    def mock_bs_roformer_model(self):
        """Create a mock BSRoformer model."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            mock_state = {
                'config': {
                    'dim': 512, 'depth': 6, 'num_stems': 2,
                    'freqs_per_bands': (2, 4, 8, 16, 32, 64),  # BSRoformer specific
                    'model_type': 'bs_roformer'
                }
            }
            torch.save(mock_state, tmp.name)
            yield tmp.name
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.fixture
    def mock_mel_band_roformer_model(self):
        """Create a mock MelBandRoformer model."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            mock_state = {
                'config': {
                    'dim': 384, 'depth': 8, 'num_stems': 4,
                    'num_bands': 64,  # MelBandRoformer specific
                    'sample_rate': 44100,
                    'model_type': 'mel_band_roformer'
                }
            }
            torch.save(mock_state, tmp.name)
            yield tmp.name
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Implementation not available yet (TDD)")
    def test_switch_bs_to_mel_band_roformer(self, mock_bs_roformer_model, mock_mel_band_roformer_model):
        """Test switching from BSRoformer to MelBandRoformer."""
        
        with tempfile.TemporaryDirectory() as output_dir:
            separator = Separator(output_dir=output_dir)
            
            # Load BSRoformer first
            separator.load_model(mock_bs_roformer_model)
            assert separator.model_name == "bs_roformer"
            
            # Switch to MelBandRoformer
            separator.load_model(mock_mel_band_roformer_model)
            assert separator.model_name == "mel_band_roformer"
            
            # Both should work without conflicts
            assert separator.model is not None
    
    def test_no_implementation_exists_yet(self):
        """TDD verification: This test ensures no implementation exists yet."""
        with pytest.raises(ImportError):
            from audio_separator.separator.roformer.roformer_loader import RoformerLoader


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
