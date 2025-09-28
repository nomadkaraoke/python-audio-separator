"""
Unit tests for ModelConfiguration validation.
Tests the ModelConfiguration dataclass and its validation logic.
"""

import pytest
from dataclasses import FrozenInstanceError

# Add the roformer module to path for imports
import sys
import os
# Find project root dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
# Go up until we find the project root (contains audio_separator/ directory)
while project_root and not os.path.exists(os.path.join(project_root, 'audio_separator')):
    parent = os.path.dirname(project_root)
    if parent == project_root:  # Reached filesystem root
        break
    project_root = parent

if project_root:
    sys.path.append(project_root)

from audio_separator.separator.roformer.model_configuration import ModelConfiguration


class TestModelConfiguration:
    """Test cases for ModelConfiguration dataclass."""
    
    def test_model_configuration_creation_valid(self):
        """Test creating a valid ModelConfiguration."""
        config = ModelConfiguration(
            dim=512,
            depth=12,
            stereo=False,
            num_stems=2,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            mlp_expansion_factor=4,
            sage_attention=False,
            zero_dc=True,
            use_torch_checkpoint=False,
            skip_connection=False
        )
        
        assert config.dim == 512
        assert config.depth == 12
        assert config.stereo is False
        assert config.num_stems == 2
        assert config.mlp_expansion_factor == 4
        assert config.sage_attention is False
    
    def test_model_configuration_defaults(self):
        """Test ModelConfiguration with minimal required parameters."""
        config = ModelConfiguration(
            dim=256,
            depth=6
        )
        
        # Check defaults are applied
        assert config.dim == 256
        assert config.depth == 6
        assert config.stereo is False  # Default
        assert config.num_stems == 1  # Default
        assert config.time_transformer_depth == 2  # Default
        assert config.freq_transformer_depth == 2  # Default
        assert config.dim_head == 64  # Default
        assert config.heads == 8  # Default
        assert config.attn_dropout == 0.0  # Default
        assert config.ff_dropout == 0.0  # Default
        assert config.flash_attn is True  # Default
        assert config.mlp_expansion_factor == 4  # Default
        assert config.sage_attention is False  # Default
        assert config.zero_dc is True  # Default
        assert config.use_torch_checkpoint is False  # Default
        assert config.skip_connection is False  # Default
    
    def test_model_configuration_immutable(self):
        """Test that ModelConfiguration is immutable (frozen)."""
        config = ModelConfiguration(dim=512, depth=12)
        
        with pytest.raises(FrozenInstanceError):
            config.dim = 1024
        
        with pytest.raises(FrozenInstanceError):
            config.depth = 24
    
    def test_model_configuration_type_validation(self):
        """Test type validation in ModelConfiguration."""
        # Valid types should work
        config = ModelConfiguration(
            dim=512,
            depth=12,
            stereo=True,
            attn_dropout=0.1,
            flash_attn=False
        )
        assert isinstance(config.dim, int)
        assert isinstance(config.stereo, bool)
        assert isinstance(config.attn_dropout, float)
    
    def test_model_configuration_edge_values(self):
        """Test edge values for ModelConfiguration."""
        # Test minimum values
        config_min = ModelConfiguration(
            dim=1,
            depth=1,
            num_stems=1,
            heads=1,
            attn_dropout=0.0,
            ff_dropout=0.0
        )
        assert config_min.dim == 1
        assert config_min.depth == 1
        assert config_min.num_stems == 1
        
        # Test larger values
        config_max = ModelConfiguration(
            dim=8192,
            depth=64,
            num_stems=16,
            heads=64,
            attn_dropout=1.0,
            ff_dropout=1.0,
            mlp_expansion_factor=16
        )
        assert config_max.dim == 8192
        assert config_max.depth == 64
        assert config_max.mlp_expansion_factor == 16
    
    def test_model_configuration_boolean_parameters(self):
        """Test boolean parameter handling."""
        config = ModelConfiguration(
            dim=512,
            depth=12,
            stereo=True,
            flash_attn=False,
            sage_attention=True,
            zero_dc=False,
            use_torch_checkpoint=True,
            skip_connection=True
        )
        
        assert config.stereo is True
        assert config.flash_attn is False
        assert config.sage_attention is True
        assert config.zero_dc is False
        assert config.use_torch_checkpoint is True
        assert config.skip_connection is True
    
    def test_model_configuration_new_parameters(self):
        """Test the new parameters added for updated Roformer implementation."""
        config = ModelConfiguration(
            dim=512,
            depth=12,
            mlp_expansion_factor=8,
            sage_attention=True,
            zero_dc=False,
            use_torch_checkpoint=True,
            skip_connection=True
        )
        
        # Verify new parameters are stored correctly
        assert config.mlp_expansion_factor == 8
        assert config.sage_attention is True
        assert config.zero_dc is False
        assert config.use_torch_checkpoint is True
        assert config.skip_connection is True
    
    def test_model_configuration_repr(self):
        """Test string representation of ModelConfiguration."""
        config = ModelConfiguration(dim=512, depth=12)
        repr_str = repr(config)
        
        assert "ModelConfiguration" in repr_str
        assert "dim=512" in repr_str
        assert "depth=12" in repr_str
    
    def test_model_configuration_equality(self):
        """Test equality comparison of ModelConfiguration instances."""
        config1 = ModelConfiguration(dim=512, depth=12, stereo=True)
        config2 = ModelConfiguration(dim=512, depth=12, stereo=True)
        config3 = ModelConfiguration(dim=512, depth=12, stereo=False)
        
        assert config1 == config2
        assert config1 != config3
    
    def test_model_configuration_hash(self):
        """Test that ModelConfiguration is hashable."""
        config1 = ModelConfiguration(dim=512, depth=12)
        config2 = ModelConfiguration(dim=512, depth=12)
        config3 = ModelConfiguration(dim=256, depth=6)
        
        # Same configurations should have same hash
        assert hash(config1) == hash(config2)
        
        # Different configurations should have different hashes
        assert hash(config1) != hash(config3)
        
        # Should be usable as dict keys
        config_dict = {config1: "first", config3: "second"}
        assert config_dict[config2] == "first"  # config2 == config1
    
    def test_model_configuration_from_dict(self):
        """Test creating ModelConfiguration from dictionary-like data."""
        data = {
            'dim': 512,
            'depth': 12,
            'stereo': True,
            'mlp_expansion_factor': 8,
            'sage_attention': True
        }
        
        config = ModelConfiguration(**data)
        
        assert config.dim == 512
        assert config.depth == 12
        assert config.stereo is True
        assert config.mlp_expansion_factor == 8
        assert config.sage_attention is True
    
    def test_model_configuration_with_extra_kwargs(self):
        """Test ModelConfiguration ignores extra unknown parameters."""
        # This should not raise an error, unknown params should be ignored
        # due to the dataclass design
        try:
            config = ModelConfiguration(
                dim=512,
                depth=12,
                unknown_param="should_be_ignored"  # This will cause TypeError
            )
            # If we get here, the dataclass accepted unknown params (unexpected)
            assert False, "Expected TypeError for unknown parameter"
        except TypeError as e:
            # This is expected - dataclasses don't accept unknown parameters
            assert "unknown_param" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])
