"""
BSRoformer-specific parameter validator.
Extends the base ParameterValidator with BSRoformer-specific validation logic.
"""

from typing import Dict, Any, List, Tuple
from .parameter_validator import ParameterValidator, ValidationIssue, ValidationSeverity


class BSRoformerValidator(ParameterValidator):
    """
    Specialized validator for BSRoformer model parameters.
    
    Provides BSRoformer-specific validation beyond the base parameter validation,
    including frequency band configuration and STFT parameter validation.
    """
    
    # BSRoformer-specific parameter constraints
    DEFAULT_FREQS_PER_BANDS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    MIN_BANDS = 2
    MAX_BANDS = 32
    
    def validate_freqs_per_bands(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate freqs_per_bands parameter for BSRoformer.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for freqs_per_bands
        """
        issues = []
        
        if 'freqs_per_bands' not in config:
            return issues  # Required parameter check handled by base class
        
        freqs_per_bands = config['freqs_per_bands']
        
        # Type check
        if not isinstance(freqs_per_bands, (list, tuple)):
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="freqs_per_bands",
                message="freqs_per_bands must be a list or tuple",
                suggested_fix="Convert freqs_per_bands to a list or tuple of integers",
                current_value=freqs_per_bands,
                expected_value="list or tuple of integers"
            )
            issues.append(issue)
            return issues
        
        # Length check
        if len(freqs_per_bands) < self.MIN_BANDS:
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="freqs_per_bands",
                message=f"freqs_per_bands must have at least {self.MIN_BANDS} bands",
                suggested_fix=f"Add more frequency bands to reach minimum of {self.MIN_BANDS}",
                current_value=len(freqs_per_bands),
                expected_value=f">= {self.MIN_BANDS}"
            )
            issues.append(issue)
        
        if len(freqs_per_bands) > self.MAX_BANDS:
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                parameter_name="freqs_per_bands",
                message=f"freqs_per_bands has {len(freqs_per_bands)} bands, which may impact performance",
                suggested_fix=f"Consider reducing to {self.MAX_BANDS} or fewer bands for better performance",
                current_value=len(freqs_per_bands),
                expected_value=f"<= {self.MAX_BANDS} (recommended)"
            )
            issues.append(issue)
        
        # Value validation
        for i, freq_count in enumerate(freqs_per_bands):
            if not isinstance(freq_count, int) or freq_count <= 0:
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    parameter_name=f"freqs_per_bands[{i}]",
                    message=f"Each frequency count must be a positive integer, got {freq_count}",
                    suggested_fix="Use positive integers for all frequency counts",
                    current_value=freq_count,
                    expected_value="positive integer"
                )
                issues.append(issue)
        
        # Check for reasonable progression (powers of 2 are common)
        if all(isinstance(f, int) and f > 0 for f in freqs_per_bands):
            # Check if values follow a reasonable pattern (not strictly enforced)
            total_freqs = sum(freqs_per_bands)
            if total_freqs > 8192:  # Unusually high for typical STFT
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    parameter_name="freqs_per_bands",
                    message=f"Total frequency count ({total_freqs}) is very high",
                    suggested_fix="Consider using fewer total frequencies for typical audio processing",
                    current_value=total_freqs,
                    expected_value="<= 8192 (typical)"
                )
                issues.append(issue)
        
        return issues
    
    def validate_stft_compatibility(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate STFT-related parameters for compatibility with freqs_per_bands.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for STFT compatibility
        """
        issues = []
        
        if 'freqs_per_bands' not in config:
            return issues
        
        freqs_per_bands = config.get('freqs_per_bands')
        stft_n_fft = config.get('stft_n_fft', 2048)  # Common default
        
        if isinstance(freqs_per_bands, (list, tuple)) and all(isinstance(f, int) for f in freqs_per_bands):
            total_freqs = sum(freqs_per_bands)
            expected_freqs = stft_n_fft // 2 + 1
            
            if total_freqs != expected_freqs:
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    parameter_name="freqs_per_bands, stft_n_fft",
                    message=f"freqs_per_bands sum ({total_freqs}) doesn't match STFT frequency bins ({expected_freqs})",
                    suggested_fix=f"Adjust freqs_per_bands to sum to {expected_freqs} or modify stft_n_fft",
                    current_value=f"sum={total_freqs}, expected={expected_freqs}",
                    expected_value=f"freqs_per_bands sum == stft_n_fft//2 + 1"
                )
                issues.append(issue)
        
        return issues
    
    def validate_mask_estimator_config(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate mask estimator configuration for BSRoformer.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for mask estimator
        """
        issues = []
        
        mask_depth = config.get('mask_estimator_depth', 2)
        
        if not isinstance(mask_depth, int) or mask_depth < 1:
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="mask_estimator_depth",
                message="mask_estimator_depth must be a positive integer",
                suggested_fix="Set mask_estimator_depth to a positive integer (typically 2-8)",
                current_value=mask_depth,
                expected_value="positive integer"
            )
            issues.append(issue)
        elif mask_depth > 8:
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                parameter_name="mask_estimator_depth",
                message=f"mask_estimator_depth of {mask_depth} may be unnecessarily deep",
                suggested_fix="Consider using a smaller depth (2-8) for better efficiency",
                current_value=mask_depth,
                expected_value="2-8 (recommended)"
            )
            issues.append(issue)
        
        return issues
    
    def validate_all(self, config: Dict[str, Any], model_type: str = "bs_roformer") -> List[ValidationIssue]:
        """
        Run all BSRoformer-specific validation checks.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model (should be "bs_roformer")
            
        Returns:
            List of all validation issues found
        """
        # Start with base validation
        all_issues = super().validate_all(config, model_type)
        
        # Add BSRoformer-specific validation
        all_issues.extend(self.validate_freqs_per_bands(config))
        all_issues.extend(self.validate_stft_compatibility(config))
        all_issues.extend(self.validate_mask_estimator_config(config))
        
        return all_issues
    
    def get_parameter_defaults(self, model_type: str = "bs_roformer") -> Dict[str, Any]:
        """
        Get BSRoformer-specific parameter defaults.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of parameter names to default values
        """
        defaults = super().get_parameter_defaults(model_type)
        
        # Override with BSRoformer-specific defaults
        defaults.update({
            'freqs_per_bands': self.DEFAULT_FREQS_PER_BANDS[:6],  # Use first 6 bands as default
            'mask_estimator_depth': 2,
            'stft_n_fft': 2048,
            'stft_hop_length': 512,
            'stft_win_length': 2048,
            'multi_stft_resolution_loss_weight': 1.0,
        })
        
        return defaults
