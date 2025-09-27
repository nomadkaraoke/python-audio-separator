"""
Unit tests for ConfigurationNormalizer methods.
Tests the configuration normalization and validation logic.
"""

import pytest
from unittest.mock import Mock, patch

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

from audio_separator.separator.roformer.configuration_normalizer import ConfigurationNormalizer
from audio_separator.separator.roformer.parameter_validation_error import ParameterValidationError


class TestConfigurationNormalizer:
    """Test cases for ConfigurationNormalizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ConfigurationNormalizer()
    
    def test_normalize_config_basic(self):
        """Test basic configuration normalization."""
        config = {
            'dim': 512,
            'depth': 12,
            'freqs_per_bands': (2, 4, 8, 16, 32, 64)
        }
        
        result = self.normalizer.normalize_config(config, "bs_roformer", apply_defaults=False, validate=False)
        
        assert result['dim'] == 512
        assert result['depth'] == 12
        assert result['freqs_per_bands'] == (2, 4, 8, 16, 32, 64)
    
    def test_normalize_config_with_defaults(self):
        """Test configuration normalization with defaults applied."""
        config = {
            'dim': 512,
            'depth': 12,
            'freqs_per_bands': (2, 4, 8, 16, 32, 64)
        }
        
        result = self.normalizer.normalize_config(config, "bs_roformer", apply_defaults=True, validate=False)
        
        # Original values preserved
        assert result['dim'] == 512
        assert result['depth'] == 12
        assert result['freqs_per_bands'] == (2, 4, 8, 16, 32, 64)
        
        # Defaults applied
        assert result['stereo'] is False
        assert result['num_stems'] == 2
        assert result['flash_attn'] is True
        assert result['mlp_expansion_factor'] == 4
    
    def test_normalize_config_with_validation_valid(self):
        """Test configuration normalization with validation - valid config."""
        config = {
            'dim': 512,
            'depth': 12,
            'freqs_per_bands': (2, 4, 8, 16, 32, 64)
        }
        
        # Should not raise any exception
        result = self.normalizer.normalize_config(config, "bs_roformer", apply_defaults=True, validate=True)
        assert result is not None
    
    def test_normalize_config_with_validation_invalid(self):
        """Test configuration normalization with validation - invalid config."""
        config = {
            'dim': "invalid",  # Invalid type
            'depth': 12
            # Missing required 'freqs_per_bands'
        }
        
        with pytest.raises(ParameterValidationError):
            self.normalizer.normalize_config(config, "bs_roformer", apply_defaults=True, validate=True)
    
    def test_normalize_structure_flat_config(self):
        """Test structure normalization with flat configuration."""
        config = {
            'dim': 512,
            'depth': 12,
            'sample_rate': 44100
        }
        
        result = self.normalizer._normalize_structure(config, "bs_roformer")
        
        assert result['dim'] == 512
        assert result['depth'] == 12
        assert result['sample_rate'] == 44100
    
    def test_normalize_structure_nested_config(self):
        """Test structure normalization with nested configuration."""
        config = {
            'model': {
                'dim': 512,
                'depth': 12
            },
            'training': {
                'sample_rate': 44100,
                'hop_length': 512
            },
            'inference': {
                'dim_t': 1024,
                'n_fft': 2048
            }
        }
        
        result = self.normalizer._normalize_structure(config, "bs_roformer")
        
        # Flattened model parameters
        assert result['dim'] == 512
        assert result['depth'] == 12
        
        # Extracted training/inference parameters
        assert result['sample_rate'] == 44100
        assert result['hop_length'] == 512
        assert result['dim_t'] == 1024
        assert result['n_fft'] == 2048
    
    def test_normalize_parameter_names_aliases(self):
        """Test parameter name normalization with aliases."""
        config = {
            'n_fft': 2048,  # Should become 'stft_n_fft'
            'hop_length': 512,  # Should become 'stft_hop_length'
            'n_heads': 8,  # Should become 'heads'
            'expansion_factor': 4,  # Should become 'mlp_expansion_factor'
            'freq_bands': (2, 4, 8, 16),  # Should become 'freqs_per_bands'
            'n_mels': 64  # Should become 'num_bands'
        }
        
        result = self.normalizer._normalize_parameter_names(config)
        
        assert result['stft_n_fft'] == 2048
        assert result['stft_hop_length'] == 512
        assert result['heads'] == 8
        assert result['mlp_expansion_factor'] == 4
        assert result['freqs_per_bands'] == (2, 4, 8, 16)
        assert result['num_bands'] == 64
        
        # Original names should not be present
        assert 'n_fft' not in result
        assert 'hop_length' not in result
        assert 'n_heads' not in result
    
    def test_normalize_parameter_values_booleans(self):
        """Test parameter value normalization for booleans."""
        config = {
            'stereo': 'true',
            'flash_attn': 'false',
            'sage_attention': '1',
            'zero_dc': 'yes',
            'use_torch_checkpoint': 'on',
            'skip_connection': '0'
        }
        
        result = self.normalizer._normalize_parameter_values(config, "bs_roformer")
        
        assert result['stereo'] is True
        assert result['flash_attn'] is False
        assert result['sage_attention'] is True
        assert result['zero_dc'] is True
        assert result['use_torch_checkpoint'] is True
        assert result['skip_connection'] is False
    
    def test_normalize_parameter_values_numbers(self):
        """Test parameter value normalization for numbers."""
        config = {
            'dim': '512',
            'depth': '12.0',  # Float string to int
            'sample_rate': 44100.0,  # Float to int
            'attn_dropout': '0.1',
            'ff_dropout': 0.2
        }
        
        result = self.normalizer._normalize_parameter_values(config, "bs_roformer")
        
        assert result['dim'] == 512
        assert result['depth'] == 12
        assert result['sample_rate'] == 44100
        assert result['attn_dropout'] == 0.1
        assert result['ff_dropout'] == 0.2
        
        # Check types
        assert isinstance(result['dim'], int)
        assert isinstance(result['depth'], int)
        assert isinstance(result['sample_rate'], int)
        assert isinstance(result['attn_dropout'], float)
        assert isinstance(result['ff_dropout'], float)
    
    def test_normalize_parameter_values_tuples(self):
        """Test parameter value normalization for tuples/lists."""
        config = {
            'freqs_per_bands': '[2, 4, 8, 16]',  # String representation
            'freqs_per_bands_2': [2, 4, 8, 16],  # List to tuple
            'freqs_per_bands_3': '(2,4,8,16)'  # String tuple
        }
        
        result = self.normalizer._normalize_parameter_values(config, "bs_roformer")
        
        assert result['freqs_per_bands'] == (2, 4, 8, 16)
        assert result['freqs_per_bands_2'] == (2, 4, 8, 16)
        assert result['freqs_per_bands_3'] == (2, 4, 8, 16)
        
        # Check types are tuples
        assert isinstance(result['freqs_per_bands'], tuple)
        assert isinstance(result['freqs_per_bands_2'], tuple)
        assert isinstance(result['freqs_per_bands_3'], tuple)
    
    def test_normalize_parameter_values_strings(self):
        """Test parameter value normalization for strings."""
        config = {
            'norm': 'LAYER_NORM',  # Should become lowercase
            'act': 'GELU',  # Should become lowercase
            'mel_scale': 'HTK'  # Should become lowercase
        }
        
        result = self.normalizer._normalize_parameter_values(config, "mel_band_roformer")
        
        assert result['norm'] == 'layer_norm'
        assert result['act'] == 'gelu'
        assert result['mel_scale'] == 'htk'
    
    def test_detect_model_type_bs_roformer(self):
        """Test model type detection for BSRoformer."""
        configs = [
            {'freqs_per_bands': (2, 4, 8, 16)},  # Direct indicator
            {'model_type': 'bs_roformer'},  # Explicit type
            {'type': 'BSRoformer'},  # Explicit type variant
            {'architecture': 'bs-roformer'}  # Architecture field
        ]
        
        for config in configs:
            result = self.normalizer.detect_model_type(config)
            assert result == "bs_roformer", f"Failed for config: {config}"
    
    def test_detect_model_type_mel_band_roformer(self):
        """Test model type detection for MelBandRoformer."""
        configs = [
            {'num_bands': 64},  # Direct indicator
            {'n_mels': 64},  # Alias
            {'mel_bands': 64},  # Alias
            {'model_type': 'mel_band_roformer'},  # Explicit type
            {'type': 'MelBandRoformer'},  # Explicit type variant
            {'architecture': 'mel-roformer'}  # Architecture field
        ]
        
        for config in configs:
            result = self.normalizer.detect_model_type(config)
            assert result == "mel_band_roformer", f"Failed for config: {config}"
    
    def test_detect_model_type_unknown(self):
        """Test model type detection for unknown configurations."""
        configs = [
            {},  # Empty config
            {'dim': 512, 'depth': 12},  # No specific indicators
            {'model_type': 'unknown'}  # Unknown type
        ]
        
        for config in configs:
            result = self.normalizer.detect_model_type(config)
            assert result is None, f"Should return None for config: {config}"
    
    def test_normalize_from_file_path_bs_roformer(self):
        """Test normalization with file path detection - BSRoformer."""
        config = {
            'dim': 512,
            'depth': 12
        }
        
        file_paths = [
            '/path/to/bs_roformer_model.ckpt',
            '/path/to/BS-Roformer-model.pth',
            '/path/to/model_bs_roformer.bin'
        ]
        
        for file_path in file_paths:
            result = self.normalizer.normalize_from_file_path(
                config, file_path, apply_defaults=True, validate=False
            )
            
            # Should have BSRoformer defaults
            assert 'freqs_per_bands' in result, f"Failed for path: {file_path}"
            assert 'mask_estimator_depth' in result, f"Failed for path: {file_path}"
    
    def test_normalize_from_file_path_mel_band_roformer(self):
        """Test normalization with file path detection - MelBandRoformer."""
        config = {
            'dim': 512,
            'depth': 12
        }
        
        file_paths = [
            '/path/to/mel_band_roformer_model.ckpt',
            '/path/to/MelBand-Roformer-model.pth',
            '/path/to/model_mel_roformer.bin'
        ]
        
        for file_path in file_paths:
            result = self.normalizer.normalize_from_file_path(
                config, file_path, apply_defaults=True, validate=False
            )
            
            # Should have MelBandRoformer defaults
            assert result['num_bands'] == 64, f"Failed for path: {file_path}"
    
    def test_normalize_from_file_path_default_fallback(self):
        """Test normalization with file path detection - default fallback."""
        config = {
            'dim': 512,
            'depth': 12
        }
        
        file_path = '/path/to/unknown_model.ckpt'
        
        with patch.object(self.normalizer, 'detect_model_type', return_value=None):
            result = self.normalizer.normalize_from_file_path(
                config, file_path, apply_defaults=True, validate=False
            )
            
            # Should default to BSRoformer
            assert 'freqs_per_bands' in result
    
    def test_normalization_preserves_original_config(self):
        """Test that normalization doesn't modify the original configuration."""
        original_config = {
            'dim': 512,
            'depth': 12,
            'n_fft': 2048  # Will be renamed to stft_n_fft
        }
        
        # Keep a copy to compare
        config_copy = original_config.copy()
        
        result = self.normalizer.normalize_config(original_config, "bs_roformer")
        
        # Original config should be unchanged
        assert original_config == config_copy
        assert 'n_fft' in original_config  # Original name preserved
        assert 'stft_n_fft' not in original_config
        
        # Result should have normalized names
        assert 'stft_n_fft' in result
        assert result['stft_n_fft'] == 2048
    
    def test_normalization_error_handling(self):
        """Test error handling during normalization."""
        # Test with invalid string values that can't be converted
        config = {
            'dim': 'not_a_number',
            'depth': 12,
            'freqs_per_bands': 'invalid_tuple_string'
        }
        
        # Should not crash, invalid values should be passed through for validator to catch
        result = self.normalizer.normalize_config(config, "bs_roformer", validate=False)
        
        assert result['dim'] == 'not_a_number'  # Passed through unchanged
        assert result['depth'] == 12  # Valid value normalized
        assert result['freqs_per_bands'] == 'invalid_tuple_string'  # Passed through unchanged
    
    def test_comprehensive_normalization_workflow(self):
        """Test complete normalization workflow with complex configuration."""
        config = {
            'model': {
                'dim': '512',
                'depth': '12.0',
                'n_heads': '8'
            },
            'training': {
                'sample_rate': 44100,
                'n_fft': 2048
            },
            'stereo': 'true',
            'flash_attn': 'false',
            'freq_bands': '[2, 4, 8, 16, 32, 64]',
            'norm': 'LAYER_NORM'
        }
        
        result = self.normalizer.normalize_config(config, "bs_roformer", apply_defaults=True, validate=False)
        
        # Structure flattened
        assert result['dim'] == 512
        assert result['depth'] == 12
        assert result['sample_rate'] == 44100
        
        # Names normalized
        assert result['heads'] == 8
        assert result['stft_n_fft'] == 2048
        assert result['freqs_per_bands'] == (2, 4, 8, 16, 32, 64)
        
        # Values normalized
        assert result['stereo'] is True
        assert result['flash_attn'] is False
        assert result['norm'] == 'layer_norm'
        
        # Defaults applied
        assert result['num_stems'] == 2
        assert result['mlp_expansion_factor'] == 4


if __name__ == "__main__":
    pytest.main([__file__])
