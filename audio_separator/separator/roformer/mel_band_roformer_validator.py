"""
MelBandRoformer-specific parameter validator.
Extends the base ParameterValidator with MelBandRoformer-specific validation logic.
"""

from typing import Dict, Any, List
from .parameter_validator import ParameterValidator, ValidationIssue, ValidationSeverity


class MelBandRoformerValidator(ParameterValidator):
    """
    Specialized validator for MelBandRoformer model parameters.
    
    Provides MelBandRoformer-specific validation beyond the base parameter validation,
    including mel-scale frequency band configuration and sample rate validation.
    """
    
    # MelBandRoformer-specific parameter constraints
    MIN_NUM_BANDS = 8
    MAX_NUM_BANDS = 512
    RECOMMENDED_NUM_BANDS = 64
    
    # Common sample rates and their typical mel band counts
    SAMPLE_RATE_BAND_RECOMMENDATIONS = {
        8000: (8, 32),
        16000: (16, 64),
        22050: (24, 80),
        44100: (32, 128),
        48000: (32, 128),
        96000: (64, 256),
    }
    
    def validate_num_bands(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate num_bands parameter for MelBandRoformer.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for num_bands
        """
        issues = []
        
        if 'num_bands' not in config:
            return issues  # Required parameter check handled by base class
        
        num_bands = config['num_bands']
        
        # Type check
        if not isinstance(num_bands, int):
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="num_bands",
                message="num_bands must be an integer",
                suggested_fix="Set num_bands to an integer value",
                current_value=num_bands,
                expected_value="integer"
            )
            issues.append(issue)
            return issues
        
        # Range check
        if num_bands < self.MIN_NUM_BANDS:
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="num_bands",
                message=f"num_bands must be at least {self.MIN_NUM_BANDS}",
                suggested_fix=f"Set num_bands to {self.MIN_NUM_BANDS} or higher",
                current_value=num_bands,
                expected_value=f">= {self.MIN_NUM_BANDS}"
            )
            issues.append(issue)
        
        if num_bands > self.MAX_NUM_BANDS:
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                parameter_name="num_bands",
                message=f"num_bands of {num_bands} is very high and may impact performance",
                suggested_fix=f"Consider using {self.MAX_NUM_BANDS} or fewer bands for better performance",
                current_value=num_bands,
                expected_value=f"<= {self.MAX_NUM_BANDS} (recommended)"
            )
            issues.append(issue)
        
        # Check if it's a power of 2 (often preferred for efficiency)
        if num_bands > 0 and (num_bands & (num_bands - 1)) != 0:
            # Find nearest powers of 2
            lower_power = 1 << (num_bands.bit_length() - 1)
            upper_power = 1 << num_bands.bit_length()
            
            issue = ValidationIssue(
                severity=ValidationSeverity.INFO,
                parameter_name="num_bands",
                message=f"num_bands ({num_bands}) is not a power of 2, which may be less efficient",
                suggested_fix=f"Consider using {lower_power} or {upper_power} for potentially better performance",
                current_value=num_bands,
                expected_value="power of 2 (optional optimization)"
            )
            issues.append(issue)
        
        return issues
    
    def validate_sample_rate_compatibility(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate sample rate compatibility with num_bands.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for sample rate compatibility
        """
        issues = []
        
        num_bands = config.get('num_bands')
        sample_rate = config.get('sample_rate')
        
        if not isinstance(num_bands, int) or not isinstance(sample_rate, int):
            return issues  # Type validation handled elsewhere
        
        # Check against known good combinations
        if sample_rate in self.SAMPLE_RATE_BAND_RECOMMENDATIONS:
            min_bands, max_bands = self.SAMPLE_RATE_BAND_RECOMMENDATIONS[sample_rate]
            
            if num_bands < min_bands:
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    parameter_name="num_bands, sample_rate",
                    message=f"num_bands ({num_bands}) may be too low for sample_rate ({sample_rate})",
                    suggested_fix=f"Consider using {min_bands}-{max_bands} bands for {sample_rate}Hz",
                    current_value=f"bands={num_bands}, sr={sample_rate}",
                    expected_value=f"{min_bands}-{max_bands} bands"
                )
                issues.append(issue)
            elif num_bands > max_bands:
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    parameter_name="num_bands, sample_rate",
                    message=f"num_bands ({num_bands}) may be too high for sample_rate ({sample_rate})",
                    suggested_fix=f"Consider using {min_bands}-{max_bands} bands for {sample_rate}Hz",
                    current_value=f"bands={num_bands}, sr={sample_rate}",
                    expected_value=f"{min_bands}-{max_bands} bands"
                )
                issues.append(issue)
        
        # General heuristic: num_bands should be much smaller than nyquist frequency
        nyquist = sample_rate // 2
        if num_bands > nyquist // 100:  # Very rough heuristic
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                parameter_name="num_bands, sample_rate",
                message=f"num_bands ({num_bands}) seems high relative to sample_rate ({sample_rate})",
                suggested_fix="Verify that this band count is appropriate for your mel-scale analysis",
                current_value=f"bands={num_bands}, nyquist={nyquist}",
                expected_value="num_bands << nyquist_frequency"
            )
            issues.append(issue)
        
        return issues
    
    def validate_mel_scale_config(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate mel-scale specific configuration parameters.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for mel-scale configuration
        """
        issues = []
        
        # Check for mel-scale related parameters if present
        mel_params = ['fmin', 'fmax', 'mel_scale']
        
        fmin = config.get('fmin', 0)
        fmax = config.get('fmax')
        sample_rate = config.get('sample_rate', 44100)
        
        if fmin is not None and not isinstance(fmin, (int, float)):
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="fmin",
                message="fmin must be a number",
                suggested_fix="Set fmin to a numeric value (typically 0-100 Hz)",
                current_value=fmin,
                expected_value="number"
            )
            issues.append(issue)
        elif isinstance(fmin, (int, float)) and fmin < 0:
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                parameter_name="fmin",
                message="fmin must be non-negative",
                suggested_fix="Set fmin to 0 or a positive frequency value",
                current_value=fmin,
                expected_value=">= 0"
            )
            issues.append(issue)
        
        if fmax is not None:
            if not isinstance(fmax, (int, float)):
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    parameter_name="fmax",
                    message="fmax must be a number",
                    suggested_fix="Set fmax to a numeric value or None for automatic setting",
                    current_value=fmax,
                    expected_value="number or None"
                )
                issues.append(issue)
            elif isinstance(fmax, (int, float)):
                if fmax <= 0:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        parameter_name="fmax",
                        message="fmax must be positive",
                        suggested_fix="Set fmax to a positive frequency value",
                        current_value=fmax,
                        expected_value="> 0"
                    )
                    issues.append(issue)
                elif fmax > sample_rate // 2:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        parameter_name="fmax, sample_rate",
                        message=f"fmax ({fmax}) exceeds Nyquist frequency ({sample_rate//2})",
                        suggested_fix=f"Set fmax to {sample_rate//2} or lower",
                        current_value=fmax,
                        expected_value=f"<= {sample_rate//2}"
                    )
                    issues.append(issue)
                elif isinstance(fmin, (int, float)) and fmax <= fmin:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        parameter_name="fmin, fmax",
                        message=f"fmax ({fmax}) must be greater than fmin ({fmin})",
                        suggested_fix="Set fmax to a value higher than fmin",
                        current_value=f"fmin={fmin}, fmax={fmax}",
                        expected_value="fmax > fmin"
                    )
                    issues.append(issue)
        
        return issues
    
    def validate_all(self, config: Dict[str, Any], model_type: str = "mel_band_roformer") -> List[ValidationIssue]:
        """
        Run all MelBandRoformer-specific validation checks.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model (should be "mel_band_roformer")
            
        Returns:
            List of all validation issues found
        """
        # Start with base validation
        all_issues = super().validate_all(config, model_type)
        
        # Add MelBandRoformer-specific validation
        all_issues.extend(self.validate_num_bands(config))
        all_issues.extend(self.validate_sample_rate_compatibility(config))
        all_issues.extend(self.validate_mel_scale_config(config))
        
        return all_issues
    
    def get_parameter_defaults(self, model_type: str = "mel_band_roformer") -> Dict[str, Any]:
        """
        Get MelBandRoformer-specific parameter defaults.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of parameter names to default values
        """
        defaults = super().get_parameter_defaults(model_type)
        
        # Override with MelBandRoformer-specific defaults
        defaults.update({
            'num_bands': self.RECOMMENDED_NUM_BANDS,
            'fmin': 0,
            'fmax': None,  # Will be set to sample_rate // 2 automatically
            'mel_scale': 'htk',  # or 'slaney'
        })
        
        return defaults
