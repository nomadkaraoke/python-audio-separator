"""
Unit tests for ParameterValidator methods.
Tests the core parameter validation logic for Roformer models.
"""

import pytest
from unittest.mock import Mock, patch

# Add the roformer module to path for imports
import sys
import os
sys.path.append('/Users/andrew/Projects/python-audio-separator')

from audio_separator.separator.roformer.parameter_validator import ParameterValidator, ValidationSeverity
from audio_separator.separator.roformer.parameter_validation_error import ParameterValidationError


class TestParameterValidator:
    """Test cases for ParameterValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()
    
    def test_validate_required_parameters_bs_roformer_valid(self):
        """Test validation of required parameters for BSRoformer - valid case."""
        config = {
            'dim': 512,
            'depth': 12,
            'freqs_per_bands': (2, 4, 8, 16, 32, 64)
        }
        
        issues = self.validator.validate_required_parameters(config, "bs_roformer")
        assert len(issues) == 0
    
    def test_validate_required_parameters_bs_roformer_missing(self):
        """Test validation of required parameters for BSRoformer - missing parameters."""
        config = {
            'dim': 512
            # Missing 'depth' and 'freqs_per_bands'
        }
        
        issues = self.validator.validate_required_parameters(config, "bs_roformer")
        assert len(issues) == 2
        
        # Check that both missing parameters are reported
        missing_params = [issue.parameter_name for issue in issues]
        assert 'depth' in missing_params
        assert 'freqs_per_bands' in missing_params
        
        # Check error severity
        for issue in issues:
            assert issue.severity == ValidationSeverity.ERROR
    
    def test_validate_required_parameters_mel_band_roformer_valid(self):
        """Test validation of required parameters for MelBandRoformer - valid case."""
        config = {
            'dim': 512,
            'depth': 12,
            'num_bands': 64
        }
        
        issues = self.validator.validate_required_parameters(config, "mel_band_roformer")
        assert len(issues) == 0
    
    def test_validate_required_parameters_mel_band_roformer_missing(self):
        """Test validation of required parameters for MelBandRoformer - missing parameters."""
        config = {
            'dim': 512
            # Missing 'depth' and 'num_bands'
        }
        
        issues = self.validator.validate_required_parameters(config, "mel_band_roformer")
        assert len(issues) == 2
        
        missing_params = [issue.parameter_name for issue in issues]
        assert 'depth' in missing_params
        assert 'num_bands' in missing_params
    
    def test_validate_parameter_types_valid(self):
        """Test parameter type validation - valid types."""
        config = {
            'dim': 512,
            'depth': 12,
            'stereo': False,
            'attn_dropout': 0.1,
            'freqs_per_bands': (2, 4, 8, 16),
            'sample_rate': 44100
        }
        
        issues = self.validator.validate_parameter_types(config)
        assert len(issues) == 0
    
    def test_validate_parameter_types_invalid(self):
        """Test parameter type validation - invalid types."""
        config = {
            'dim': "512",  # Should be int
            'stereo': "false",  # Should be bool
            'attn_dropout': "0.1",  # Should be float
            'sample_rate': 44100.5  # Should be int
        }
        
        issues = self.validator.validate_parameter_types(config)
        assert len(issues) == 4
        
        # Check that all type errors are reported
        invalid_params = [issue.parameter_name for issue in issues]
        assert 'dim' in invalid_params
        assert 'stereo' in invalid_params
        assert 'attn_dropout' in invalid_params
        assert 'sample_rate' in invalid_params
    
    def test_validate_parameter_ranges_valid(self):
        """Test parameter range validation - valid ranges."""
        config = {
            'dim': 512,
            'depth': 12,
            'heads': 8,
            'attn_dropout': 0.1,
            'sample_rate': 44100
        }
        
        issues = self.validator.validate_parameter_ranges(config)
        assert len(issues) == 0
    
    def test_validate_parameter_ranges_invalid(self):
        """Test parameter range validation - invalid ranges."""
        config = {
            'dim': 0,  # Below minimum (1)
            'depth': 100,  # Above maximum (64)
            'attn_dropout': 1.5,  # Above maximum (1.0)
            'sample_rate': 5000  # Below minimum (8000)
        }
        
        issues = self.validator.validate_parameter_ranges(config)
        assert len(issues) == 4
        
        # Check that all range errors are reported
        invalid_params = [issue.parameter_name for issue in issues]
        assert 'dim' in invalid_params
        assert 'depth' in invalid_params
        assert 'attn_dropout' in invalid_params
        assert 'sample_rate' in invalid_params
    
    def test_validate_parameter_compatibility_sage_flash_conflict(self):
        """Test parameter compatibility - sage_attention and flash_attn conflict."""
        config = {
            'sage_attention': True,
            'flash_attn': True
        }
        
        issues = self.validator.validate_parameter_compatibility(config)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
        assert "sage_attention, flash_attn" in issues[0].parameter_name
    
    def test_validate_parameter_compatibility_freqs_per_bands_warning(self):
        """Test parameter compatibility - freqs_per_bands sum warning."""
        config = {
            'freqs_per_bands': (1, 2, 3)  # Sum = 6, which is very low
        }
        
        issues = self.validator.validate_parameter_compatibility(config)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].parameter_name == "freqs_per_bands"
    
    def test_validate_normalization_config_valid(self):
        """Test normalization configuration validation - valid cases."""
        valid_configs = [
            None,
            'layer_norm',
            'batch_norm',
            'rms_norm',
            {'type': 'layer_norm', 'eps': 1e-5}
        ]
        
        for norm_config in valid_configs:
            issues = self.validator.validate_normalization_config(norm_config)
            assert len(issues) == 0, f"Failed for config: {norm_config}"
    
    def test_validate_normalization_config_invalid(self):
        """Test normalization configuration validation - invalid cases."""
        invalid_configs = [
            'invalid_norm',  # Unsupported string
            123,  # Invalid type
            ['layer_norm']  # Invalid type (list)
        ]
        
        for norm_config in invalid_configs:
            issues = self.validator.validate_normalization_config(norm_config)
            assert len(issues) >= 1, f"Should have failed for config: {norm_config}"
            assert issues[0].severity == ValidationSeverity.ERROR
    
    def test_get_parameter_defaults_bs_roformer(self):
        """Test getting parameter defaults for BSRoformer."""
        defaults = self.validator.get_parameter_defaults("bs_roformer")
        
        # Check some expected defaults
        assert defaults['stereo'] is False
        assert defaults['num_stems'] == 2
        assert defaults['flash_attn'] is True
        assert defaults['mlp_expansion_factor'] == 4
        assert defaults['sage_attention'] is False
        assert 'freqs_per_bands' in defaults
        assert 'mask_estimator_depth' in defaults
    
    def test_get_parameter_defaults_mel_band_roformer(self):
        """Test getting parameter defaults for MelBandRoformer."""
        defaults = self.validator.get_parameter_defaults("mel_band_roformer")
        
        # Check some expected defaults
        assert defaults['stereo'] is False
        assert defaults['num_stems'] == 2
        assert defaults['flash_attn'] is True
        assert defaults['mlp_expansion_factor'] == 4
        assert defaults['sage_attention'] is False
        assert defaults['num_bands'] == 64
    
    def test_apply_parameter_defaults(self):
        """Test applying parameter defaults to configuration."""
        config = {
            'dim': 512,
            'depth': 12,
            'stereo': True  # Override default
        }
        
        result = self.validator.apply_parameter_defaults(config, "bs_roformer")
        
        # Should have original values
        assert result['dim'] == 512
        assert result['depth'] == 12
        assert result['stereo'] is True  # Override preserved
        
        # Should have applied defaults
        assert result['num_stems'] == 2
        assert result['flash_attn'] is True
        assert result['mlp_expansion_factor'] == 4
    
    def test_validate_all_comprehensive(self):
        """Test comprehensive validation with all checks."""
        config = {
            'dim': 512,
            'depth': 12,
            'freqs_per_bands': (2, 4, 8, 16, 32, 64),
            'stereo': False,
            'attn_dropout': 0.1,
            'sage_attention': False,
            'norm': 'layer_norm'
        }
        
        issues = self.validator.validate_all(config, "bs_roformer")
        assert len(issues) == 0
    
    def test_validate_all_with_errors(self):
        """Test comprehensive validation with multiple error types."""
        config = {
            'dim': "invalid",  # Type error
            'depth': 100,  # Range error
            # Missing 'freqs_per_bands' - required parameter error
            'sage_attention': True,
            'flash_attn': True,  # Compatibility warning
            'norm': 'invalid_norm'  # Normalization error
        }
        
        issues = self.validator.validate_all(config, "bs_roformer")
        assert len(issues) >= 5  # At least 5 different types of issues
        
        # Check we have different types of issues
        severities = [issue.severity for issue in issues]
        assert ValidationSeverity.ERROR in severities
        assert ValidationSeverity.WARNING in severities
    
    def test_validate_and_raise_success(self):
        """Test validate_and_raise with valid configuration."""
        config = {
            'dim': 512,
            'depth': 12,
            'freqs_per_bands': (2, 4, 8, 16, 32, 64)
        }
        
        # Should not raise any exception
        self.validator.validate_and_raise(config, "bs_roformer")
    
    def test_validate_and_raise_error(self):
        """Test validate_and_raise with invalid configuration."""
        config = {
            'dim': "invalid",  # Type error
            'depth': 12
            # Missing 'freqs_per_bands'
        }
        
        with pytest.raises(ParameterValidationError) as exc_info:
            self.validator.validate_and_raise(config, "bs_roformer")
        
        # Check the exception contains useful information
        exception = exc_info.value
        assert exception.parameter_name is not None
        assert exception.suggested_fix is not None
    
    def test_private_helper_methods(self):
        """Test private helper methods."""
        # Test _is_correct_type
        assert self.validator._is_correct_type(123, int) is True
        assert self.validator._is_correct_type("123", int) is False
        assert self.validator._is_correct_type([1, 2, 3], (list, tuple)) is True
        assert self.validator._is_correct_type((1, 2, 3), (list, tuple)) is True
        
        # Test _get_type_name
        assert self.validator._get_type_name(int) == "int"
        assert "int" in self.validator._get_type_name((int, float))
        assert "float" in self.validator._get_type_name((int, float))
        
        # Test _get_expected_type_description
        assert "int" in self.validator._get_expected_type_description('dim')
        assert "appropriate type" in self.validator._get_expected_type_description('unknown_param')


if __name__ == "__main__":
    pytest.main([__file__])
