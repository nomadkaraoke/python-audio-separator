"""
Parameter validator implementation.
Validates Roformer model parameters according to interface contracts.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import sys
import os

# Add contracts to path for interface imports (optional)
try:
    # Find project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    # Go up until we find the project root (contains specs/ directory)
    while project_root and not os.path.exists(os.path.join(project_root, 'specs')):
        parent = os.path.dirname(project_root)
        if parent == project_root:  # Reached filesystem root
            break
        project_root = parent
    
    contracts_path = os.path.join(project_root, 'specs', '001-update-roformer-implementation', 'contracts')
    if os.path.exists(contracts_path):
        sys.path.append(contracts_path)
    from parameter_validator_interface import (
        ParameterValidatorInterface,
        ValidationIssue,
        ValidationSeverity
    )
    _has_interface = True
except ImportError:
    # Create dummy interfaces for when contracts are not available
    from enum import Enum
    from dataclasses import dataclass
    
    class ValidationSeverity(Enum):
        ERROR = "error"
        WARNING = "warning"
        INFO = "info"
    
    @dataclass
    class ValidationIssue:
        severity: ValidationSeverity
        parameter_name: str
        message: str
        suggested_fix: str
        current_value: any = None
        expected_value: any = None
    
    class ParameterValidatorInterface:
        pass
    
    _has_interface = False
from .parameter_validation_error import ParameterValidationError


class ParameterValidator(ParameterValidatorInterface):
    """
    Implementation of parameter validation for Roformer models.
    
    Validates model parameters according to the interface contract,
    providing detailed error messages and suggestions for fixes.
    """
    
    # Define valid parameter types and ranges
    PARAMETER_TYPES = {
        'dim': int,
        'depth': int,
        'stereo': bool,
        'num_stems': int,
        'time_transformer_depth': int,
        'freq_transformer_depth': int,
        'dim_head': int,
        'heads': int,
        'attn_dropout': float,
        'ff_dropout': float,
        'flash_attn': bool,
        'mlp_expansion_factor': int,
        'sage_attention': bool,
        'zero_dc': bool,
        'use_torch_checkpoint': bool,
        'skip_connection': bool,
        'sample_rate': int,
        'freqs_per_bands': (tuple, list),
        'num_bands': int,
        'mask_estimator_depth': int,
    }
    
    PARAMETER_RANGES = {
        'dim': (1, 8192),
        'depth': (1, 64),
        'num_stems': (1, 16),
        'time_transformer_depth': (1, 32),
        'freq_transformer_depth': (1, 32),
        'dim_head': (1, 1024),
        'heads': (1, 64),
        'attn_dropout': (0.0, 1.0),
        'ff_dropout': (0.0, 1.0),
        'mlp_expansion_factor': (1, 16),
        'sample_rate': (8000, 192000),
        'num_bands': (8, 512),
        'mask_estimator_depth': (1, 8),
    }
    
    REQUIRED_PARAMETERS = {
        'bs_roformer': ['dim', 'depth', 'freqs_per_bands'],
        'mel_band_roformer': ['dim', 'depth', 'num_bands'],
    }
    
    SUPPORTED_NORMALIZATION_TYPES = [
        'layer_norm', 'batch_norm', 'rms_norm', 'group_norm', 
        'instance_norm', None, 'none'
    ]
    
    def validate_required_parameters(self, config: Dict[str, Any], model_type: str) -> List[ValidationIssue]:
        """
        Validate that all required parameters are present.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model ("bs_roformer" or "mel_band_roformer")
            
        Returns:
            List of validation issues for missing required parameters
        """
        issues = []
        
        required_params = self.REQUIRED_PARAMETERS.get(model_type, [])
        
        for param_name in required_params:
            if param_name not in config:
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    parameter_name=param_name,
                    message=f"Required parameter '{param_name}' is missing for {model_type}",
                    suggested_fix=f"Add '{param_name}' parameter with appropriate {self._get_expected_type_description(param_name)} value",
                    current_value=None,
                    expected_value=self._get_expected_type_description(param_name)
                )
                issues.append(issue)
        
        return issues
    
    def validate_parameter_types(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate parameter types match expected types.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for type mismatches
        """
        issues = []
        
        for param_name, value in config.items():
            if param_name in self.PARAMETER_TYPES:
                expected_type = self.PARAMETER_TYPES[param_name]
                
                if not self._is_correct_type(value, expected_type):
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        parameter_name=param_name,
                        message=f"Parameter '{param_name}' has incorrect type",
                        suggested_fix=f"Change '{param_name}' to {self._get_type_name(expected_type)}",
                        current_value=value,
                        expected_value=self._get_type_name(expected_type)
                    )
                    issues.append(issue)
        
        return issues
    
    def validate_parameter_ranges(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate parameter values are within acceptable ranges.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for out-of-range values
        """
        issues = []
        
        for param_name, value in config.items():
            if param_name in self.PARAMETER_RANGES:
                min_val, max_val = self.PARAMETER_RANGES[param_name]
                
                if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        parameter_name=param_name,
                        message=f"Parameter '{param_name}' value {value} is outside valid range [{min_val}, {max_val}]",
                        suggested_fix=f"Set '{param_name}' to a value between {min_val} and {max_val}",
                        current_value=value,
                        expected_value=f"{min_val} <= value <= {max_val}"
                    )
                    issues.append(issue)
        
        return issues
    
    def validate_parameter_compatibility(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate that parameter combinations are compatible.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for incompatible parameter combinations
        """
        issues = []
        
        # Check sage_attention and flash_attn compatibility
        if config.get('sage_attention', False) and config.get('flash_attn', False):
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                parameter_name="sage_attention, flash_attn",
                message="Using both sage_attention=True and flash_attn=True may cause conflicts",
                suggested_fix="Consider using only one attention mechanism",
                current_value="both True",
                expected_value="only one True"
            )
            issues.append(issue)
        
        # Check freqs_per_bands consistency for BSRoformer
        if 'freqs_per_bands' in config:
            freqs = config['freqs_per_bands']
            if isinstance(freqs, (list, tuple)) and len(freqs) > 0:
                total_freqs = sum(freqs)
                # Check if it looks like a reasonable STFT frequency count
                if total_freqs < 64 or total_freqs > 4096:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        parameter_name="freqs_per_bands",
                        message=f"Sum of freqs_per_bands ({total_freqs}) may be incompatible with typical STFT configurations",
                        suggested_fix="Verify that freqs_per_bands sum matches your STFT n_fft//2 + 1",
                        current_value=total_freqs,
                        expected_value="64 to 4096 (typical range)"
                    )
                    issues.append(issue)
        
        # Check num_bands vs sample_rate for MelBandRoformer
        if 'num_bands' in config and 'sample_rate' in config:
            num_bands = config['num_bands']
            sample_rate = config['sample_rate']
            if num_bands > sample_rate // 100:  # Very rough heuristic
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    parameter_name="num_bands, sample_rate",
                    message=f"num_bands ({num_bands}) may be too high for sample_rate ({sample_rate})",
                    suggested_fix="Consider reducing num_bands or verify it's appropriate for your use case",
                    current_value=f"bands={num_bands}, sr={sample_rate}",
                    expected_value="num_bands << sample_rate"
                )
                issues.append(issue)
        
        return issues
    
    def validate_normalization_config(self, norm_config: Any) -> List[ValidationIssue]:
        """
        Validate normalization configuration.
        
        Args:
            norm_config: Normalization configuration (may be string, dict, or None)
            
        Returns:
            List of validation issues for normalization configuration
        """
        issues = []
        
        if norm_config is not None:
            if isinstance(norm_config, str):
                if norm_config not in self.SUPPORTED_NORMALIZATION_TYPES:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        parameter_name="norm",
                        message=f"Unsupported normalization type '{norm_config}'",
                        suggested_fix=f"Use one of: {', '.join(str(t) for t in self.SUPPORTED_NORMALIZATION_TYPES if t is not None)}",
                        current_value=norm_config,
                        expected_value="supported normalization type"
                    )
                    issues.append(issue)
            elif not isinstance(norm_config, dict):
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    parameter_name="norm",
                    message=f"Normalization config must be string, dict, or None, got {type(norm_config).__name__}",
                    suggested_fix="Use a string normalization type or None",
                    current_value=norm_config,
                    expected_value="string, dict, or None"
                )
                issues.append(issue)
        
        return issues
    
    def get_parameter_defaults(self, model_type: str) -> Dict[str, Any]:
        """
        Get default values for optional parameters.
        
        Args:
            model_type: Type of model ("bs_roformer" or "mel_band_roformer")
            
        Returns:
            Dictionary of parameter names to default values
        """
        defaults = {
            'stereo': False,
            'num_stems': 2,
            'time_transformer_depth': 2,
            'freq_transformer_depth': 2,
            'dim_head': 64,
            'heads': 8,
            'attn_dropout': 0.0,
            'ff_dropout': 0.0,
            'flash_attn': True,
            'mlp_expansion_factor': 4,
            'sage_attention': False,
            'zero_dc': True,
            'use_torch_checkpoint': False,
            'skip_connection': False,
            'sample_rate': 44100,
            'norm': None,
        }
        
        # Add model-specific defaults
        if model_type == 'bs_roformer':
            defaults.update({
                'freqs_per_bands': (2, 4, 8, 16, 32, 64),
                'mask_estimator_depth': 2,
            })
        elif model_type == 'mel_band_roformer':
            defaults.update({
                'num_bands': 64,
            })
        
        return defaults
    
    def apply_parameter_defaults(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Apply default values to missing optional parameters.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model
            
        Returns:
            Configuration with defaults applied
        """
        defaults = self.get_parameter_defaults(model_type)
        result_config = defaults.copy()
        result_config.update(config)  # Override defaults with provided values
        
        return result_config
    
    def _is_correct_type(self, value: Any, expected_type: Union[type, Tuple[type, ...]]) -> bool:
        """Check if value matches expected type(s)."""
        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)
        return isinstance(value, expected_type)
    
    def _get_type_name(self, expected_type: Union[type, Tuple[type, ...]]) -> str:
        """Get human-readable type name."""
        if isinstance(expected_type, tuple):
            return " or ".join(t.__name__ for t in expected_type)
        return expected_type.__name__
    
    def _get_expected_type_description(self, param_name: str) -> str:
        """Get description of expected type for a parameter."""
        if param_name in self.PARAMETER_TYPES:
            return self._get_type_name(self.PARAMETER_TYPES[param_name])
        return "appropriate type"
    
    def validate_all(self, config: Dict[str, Any], model_type: str) -> List[ValidationIssue]:
        """
        Run all validation checks on a configuration.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model
            
        Returns:
            List of all validation issues found
        """
        all_issues = []
        
        all_issues.extend(self.validate_required_parameters(config, model_type))
        all_issues.extend(self.validate_parameter_types(config))
        all_issues.extend(self.validate_parameter_ranges(config))
        all_issues.extend(self.validate_parameter_compatibility(config))
        
        # Validate normalization if present
        if 'norm' in config:
            all_issues.extend(self.validate_normalization_config(config['norm']))
        
        return all_issues
    
    def validate_and_raise(self, config: Dict[str, Any], model_type: str) -> None:
        """
        Validate configuration and raise ParameterValidationError if issues found.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model
            
        Raises:
            ParameterValidationError: If validation issues are found
        """
        issues = self.validate_all(config, model_type)
        
        # Find first error (not warning)
        error_issues = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        
        if error_issues:
            first_error = error_issues[0]
            raise ParameterValidationError(
                parameter_name=first_error.parameter_name,
                expected_type=first_error.expected_value or "valid value",
                actual_value=first_error.current_value,
                suggested_fix=first_error.suggested_fix,
                context=f"{model_type} model validation"
            )
