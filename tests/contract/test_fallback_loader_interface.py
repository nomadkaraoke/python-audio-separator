"""
Contract tests for FallbackLoaderInterface.
These tests verify the fallback loading mechanism interface contracts.
"""

import pytest
from unittest.mock import Mock

# Import interfaces from contracts
import sys
sys.path.append('/Users/andrew/Projects/python-audio-separator/specs/001-update-roformer-implementation/contracts')

from roformer_loader_interface import (
    FallbackLoaderInterface,
    ModelLoadingResult,
    ModelConfiguration,
    ImplementationVersion
)


class TestFallbackLoaderInterface:
    """Test the FallbackLoader interface contract."""
    
    def test_try_new_implementation_contract(self):
        """Test try_new_implementation interface contract."""
        loader = Mock(spec=FallbackLoaderInterface)
        
        # Mock successful new implementation
        config = ModelConfiguration(dim=512, depth=6)
        expected_result = ModelLoadingResult(
            success=True,
            model=Mock(),
            implementation_used=ImplementationVersion.NEW
        )
        loader.try_new_implementation.return_value = expected_result
        
        result = loader.try_new_implementation("/path/to/model.ckpt", config)
        
        assert isinstance(result, ModelLoadingResult)
        assert result.success is True
        assert result.implementation_used == ImplementationVersion.NEW
        
        # Mock new implementation failure
        failure_result = ModelLoadingResult(
            success=False,
            error_message="TypeError: unexpected keyword argument 'mlp_expansion_factor'",
            implementation_used=ImplementationVersion.NEW
        )
        loader.try_new_implementation.return_value = failure_result
        
        result = loader.try_new_implementation("/path/to/model.ckpt", config)
        assert result.success is False
        assert "mlp_expansion_factor" in result.error_message
    
    def test_try_old_implementation_contract(self):
        """Test try_old_implementation interface contract."""
        loader = Mock(spec=FallbackLoaderInterface)
        
        # Mock successful old implementation (fallback)
        config = ModelConfiguration(dim=512, depth=6)
        expected_result = ModelLoadingResult(
            success=True,
            model=Mock(),
            implementation_used=ImplementationVersion.FALLBACK
        )
        loader.try_old_implementation.return_value = expected_result
        
        result = loader.try_old_implementation("/path/to/model.ckpt", config)
        
        assert isinstance(result, ModelLoadingResult)
        assert result.success is True
        assert result.implementation_used == ImplementationVersion.FALLBACK
    
    def test_should_fallback_contract(self):
        """Test should_fallback interface contract."""
        loader = Mock(spec=FallbackLoaderInterface)
        
        # Mock fallback decision for TypeError (should fallback)
        type_error = TypeError("BSRoformer.__init__() got an unexpected keyword argument 'mlp_expansion_factor'")
        loader.should_fallback.return_value = True
        
        should_fallback = loader.should_fallback(type_error)
        assert isinstance(should_fallback, bool)
        assert should_fallback is True
        
        # Mock no fallback for other errors
        file_error = FileNotFoundError("Model file not found")
        loader.should_fallback.return_value = False
        
        should_fallback = loader.should_fallback(file_error)
        assert should_fallback is False


# This test MUST FAIL when run - there's no real implementation yet
def test_real_fallback_loader_does_not_exist():
    """This test verifies that no real fallback loader implementation exists yet (TDD requirement)."""
    
    # Try to import the actual implementation - this should fail
    with pytest.raises(ImportError):
        from audio_separator.separator.roformer.fallback_loader import FallbackLoader
    
    # Verify the planned file doesn't exist yet
    import os
    expected_path = "/Users/andrew/Projects/python-audio-separator/audio_separator/separator/roformer/fallback_loader.py"
    assert not os.path.exists(expected_path), "Implementation file should not exist yet (TDD)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
