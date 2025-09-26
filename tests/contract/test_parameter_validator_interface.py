"""
Contract tests for ParameterValidatorInterface.
These tests verify the parameter validation interface contracts.
"""

import pytest
from unittest.mock import Mock
from typing import Dict, Any, List

# Import interfaces from contracts
import sys
sys.path.append('/Users/andrew/Projects/python-audio-separator/specs/001-update-roformer-implementation/contracts')

from parameter_validator_interface import (
    ParameterValidatorInterface,
    BSRoformerValidatorInterface,
    MelBandRoformerValidatorInterface,
    ConfigurationNormalizerInterface,
    ValidationIssue,
    ValidationSeverity
)


class TestParameterValidatorInterface:
    """Test the ParameterValidator interface contract."""
    
    def test_validate_required_parameters_contract(self):
        """Test validate_required_parameters interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock valid configuration - should return no issues
        validator.validate_required_parameters.return_value = []
        
        config = {"dim": 512, "depth": 6}
        issues = validator.validate_required_parameters(config, "bs_roformer")
        
        assert isinstance(issues, list)
        assert len(issues) == 0  # Valid config should have no issues
        
        # Mock invalid configuration - missing required parameters
        missing_param_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            parameter_name="dim",
            message="Required parameter 'dim' is missing",
            suggested_fix="Add 'dim' parameter with positive integer value",
            current_value=None,
            expected_value="positive integer"
        )
        validator.validate_required_parameters.return_value = [missing_param_issue]
        
        incomplete_config = {"depth": 6}  # Missing dim
        issues = validator.validate_required_parameters(incomplete_config, "bs_roformer")
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert all(isinstance(issue, ValidationIssue) for issue in issues)
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].parameter_name == "dim"
    
    def test_validate_parameter_types_contract(self):
        """Test validate_parameter_types interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock type validation error
        type_error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            parameter_name="dim",
            message="Parameter 'dim' must be an integer",
            suggested_fix="Change 'dim' value to an integer",
            current_value="512",  # String instead of int
            expected_value="integer"
        )
        validator.validate_parameter_types.return_value = [type_error_issue]
        
        config = {"dim": "512", "depth": 6}  # dim should be int, not string
        issues = validator.validate_parameter_types(config)
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].parameter_name == "dim"
        assert issues[0].current_value == "512"
    
    def test_validate_parameter_ranges_contract(self):
        """Test validate_parameter_ranges interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock range validation error
        range_error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            parameter_name="attn_dropout",
            message="Parameter 'attn_dropout' must be between 0.0 and 1.0",
            suggested_fix="Set 'attn_dropout' to a value between 0.0 and 1.0",
            current_value=1.5,
            expected_value="0.0 <= value <= 1.0"
        )
        validator.validate_parameter_ranges.return_value = [range_error_issue]
        
        config = {"dim": 512, "depth": 6, "attn_dropout": 1.5}  # Invalid range
        issues = validator.validate_parameter_ranges(config)
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].parameter_name == "attn_dropout"
        assert issues[0].current_value == 1.5
    
    def test_validate_parameter_compatibility_contract(self):
        """Test validate_parameter_compatibility interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock compatibility validation warning
        compatibility_issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            parameter_name="sage_attention",
            message="sage_attention=True may conflict with flash_attn=True",
            suggested_fix="Consider using only one attention mechanism",
            current_value=True,
            expected_value="False when flash_attn=True"
        )
        validator.validate_parameter_compatibility.return_value = [compatibility_issue]
        
        config = {"sage_attention": True, "flash_attn": True}
        issues = validator.validate_parameter_compatibility(config)
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert issues[0].severity == ValidationSeverity.WARNING
        assert "conflict" in issues[0].message.lower()
    
    def test_validate_normalization_config_contract(self):
        """Test validate_normalization_config interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock valid normalization config
        validator.validate_normalization_config.return_value = []
        
        issues = validator.validate_normalization_config("layer_norm")
        assert isinstance(issues, list)
        assert len(issues) == 0
        
        # Mock invalid normalization config
        norm_error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            parameter_name="norm",
            message="Unknown normalization type 'invalid_norm'",
            suggested_fix="Use one of: 'layer_norm', 'batch_norm', 'rms_norm', or None",
            current_value="invalid_norm",
            expected_value="valid normalization type"
        )
        validator.validate_normalization_config.return_value = [norm_error_issue]
        
        issues = validator.validate_normalization_config("invalid_norm")
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert issues[0].parameter_name == "norm"
    
    def test_get_parameter_defaults_contract(self):
        """Test get_parameter_defaults interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock default parameters
        default_params = {
            "mlp_expansion_factor": 4,
            "sage_attention": False,
            "zero_dc": True,
            "use_torch_checkpoint": False,
            "skip_connection": False
        }
        validator.get_parameter_defaults.return_value = default_params
        
        defaults = validator.get_parameter_defaults("bs_roformer")
        
        assert isinstance(defaults, dict)
        assert "mlp_expansion_factor" in defaults
        assert defaults["mlp_expansion_factor"] == 4
        assert isinstance(defaults["sage_attention"], bool)
    
    def test_apply_parameter_defaults_contract(self):
        """Test apply_parameter_defaults interface contract."""
        validator = Mock(spec=ParameterValidatorInterface)
        
        # Mock config with defaults applied
        input_config = {"dim": 512, "depth": 6}
        expected_config = {
            "dim": 512,
            "depth": 6,
            "mlp_expansion_factor": 4,
            "sage_attention": False,
            "zero_dc": True
        }
        validator.apply_parameter_defaults.return_value = expected_config
        
        result_config = validator.apply_parameter_defaults(input_config, "bs_roformer")
        
        assert isinstance(result_config, dict)
        assert "dim" in result_config
        assert "mlp_expansion_factor" in result_config
        assert result_config["dim"] == 512
        assert result_config["mlp_expansion_factor"] == 4


class TestBSRoformerValidatorInterface:
    """Test the BSRoformer-specific validator interface contract."""
    
    def test_validate_freqs_per_bands_contract(self):
        """Test validate_freqs_per_bands interface contract."""
        validator = Mock(spec=BSRoformerValidatorInterface)
        
        # Mock validation error for mismatched frequency bands
        freq_error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            parameter_name="freqs_per_bands",
            message="Sum of freqs_per_bands (126) does not match expected frequency bins (129)",
            suggested_fix="Adjust freqs_per_bands to sum to 129",
            current_value=(2, 4, 8, 16, 32, 64),  # sums to 126
            expected_value="tuple summing to 129"
        )
        validator.validate_freqs_per_bands.return_value = [freq_error_issue]
        
        freqs_per_bands = (2, 4, 8, 16, 32, 64)
        stft_config = {"n_fft": 256}
        issues = validator.validate_freqs_per_bands(freqs_per_bands, stft_config)
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert issues[0].parameter_name == "freqs_per_bands"
        assert "sum" in issues[0].message.lower()
    
    def test_calculate_expected_freqs_contract(self):
        """Test calculate_expected_freqs interface contract."""
        validator = Mock(spec=BSRoformerValidatorInterface)
        
        # Mock expected frequency calculation
        validator.calculate_expected_freqs.return_value = 129  # n_fft//2 + 1 for n_fft=256
        
        expected_freqs = validator.calculate_expected_freqs(256)
        
        assert isinstance(expected_freqs, int)
        assert expected_freqs > 0


class TestMelBandRoformerValidatorInterface:
    """Test the MelBandRoformer-specific validator interface contract."""
    
    def test_validate_num_bands_contract(self):
        """Test validate_num_bands interface contract."""
        validator = Mock(spec=MelBandRoformerValidatorInterface)
        
        # Mock successful validation
        validator.validate_num_bands.return_value = []
        
        issues = validator.validate_num_bands(64, 44100)
        assert isinstance(issues, list)
        assert len(issues) == 0
        
        # Mock validation error for invalid band count
        band_error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            parameter_name="num_bands",
            message="num_bands must be positive integer",
            suggested_fix="Set num_bands to a positive integer value",
            current_value=0,
            expected_value="positive integer"
        )
        validator.validate_num_bands.return_value = [band_error_issue]
        
        issues = validator.validate_num_bands(0, 44100)
        assert len(issues) > 0
        assert issues[0].parameter_name == "num_bands"
    
    def test_validate_sample_rate_contract(self):
        """Test validate_sample_rate interface contract."""
        validator = Mock(spec=MelBandRoformerValidatorInterface)
        
        # Mock sample rate validation
        validator.validate_sample_rate.return_value = []
        
        issues = validator.validate_sample_rate(44100)
        assert isinstance(issues, list)
        assert len(issues) == 0


class TestConfigurationNormalizerInterface:
    """Test the ConfigurationNormalizer interface contract."""
    
    def test_normalize_config_format_contract(self):
        """Test normalize_config_format interface contract."""
        normalizer = Mock(spec=ConfigurationNormalizerInterface)
        
        # Mock normalization from object to dict
        normalized_config = {"dim": 512, "depth": 6}
        normalizer.normalize_config_format.return_value = normalized_config
        
        # Test with mock object input
        raw_config = Mock()
        result = normalizer.normalize_config_format(raw_config)
        
        assert isinstance(result, dict)
        assert "dim" in result
        assert "depth" in result
    
    def test_map_legacy_parameters_contract(self):
        """Test map_legacy_parameters interface contract."""
        normalizer = Mock(spec=ConfigurationNormalizerInterface)
        
        # Mock legacy parameter mapping
        legacy_config = {"old_param_name": 512}
        mapped_config = {"dim": 512}  # old_param_name -> dim
        normalizer.map_legacy_parameters.return_value = mapped_config
        
        result = normalizer.map_legacy_parameters(legacy_config)
        
        assert isinstance(result, dict)
        assert "dim" in result
        assert "old_param_name" not in result
    
    def test_extract_nested_config_contract(self):
        """Test extract_nested_config interface contract."""
        normalizer = Mock(spec=ConfigurationNormalizerInterface)
        
        # Mock nested config extraction
        normalizer.extract_nested_config.return_value = "layer_norm"
        
        config = {"model": {"norm": "layer_norm"}}
        result = normalizer.extract_nested_config(config, "model.norm")
        
        assert result == "layer_norm"
        
        # Mock missing nested config
        normalizer.extract_nested_config.return_value = None
        
        result = normalizer.extract_nested_config(config, "model.missing_field")
        assert result is None


# TDD placeholder test removed - implementation is now complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
