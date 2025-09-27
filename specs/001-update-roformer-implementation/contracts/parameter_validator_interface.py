"""
API Contract: Parameter Validator Interface
This defines the interface for validating Roformer model parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Blocks model loading
    WARNING = "warning"  # Allows loading but may affect performance
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """Represents a validation issue found in model configuration."""
    severity: ValidationSeverity
    parameter_name: str
    message: str
    suggested_fix: str
    current_value: Any = None
    expected_value: Any = None


class ParameterValidatorInterface(ABC):
    """Abstract interface for validating model parameters."""
    
    @abstractmethod
    def validate_required_parameters(self, config: Dict[str, Any], model_type: str) -> List[ValidationIssue]:
        """
        Validate that all required parameters are present.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model ("bs_roformer" or "mel_band_roformer")
            
        Returns:
            List of validation issues for missing required parameters
        """
        pass
    
    @abstractmethod
    def validate_parameter_types(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate parameter types match expected types.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for type mismatches
        """
        pass
    
    @abstractmethod
    def validate_parameter_ranges(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate parameter values are within acceptable ranges.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for out-of-range values
        """
        pass
    
    @abstractmethod
    def validate_parameter_compatibility(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate that parameter combinations are compatible.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation issues for incompatible parameter combinations
        """
        pass
    
    @abstractmethod
    def validate_normalization_config(self, norm_config: Any) -> List[ValidationIssue]:
        """
        Validate normalization configuration.
        
        Args:
            norm_config: Normalization configuration (may be string, dict, or None)
            
        Returns:
            List of validation issues for normalization configuration
        """
        pass
    
    @abstractmethod
    def get_parameter_defaults(self, model_type: str) -> Dict[str, Any]:
        """
        Get default values for optional parameters.
        
        Args:
            model_type: Type of model ("bs_roformer" or "mel_band_roformer")
            
        Returns:
            Dictionary of parameter names to default values
        """
        pass
    
    @abstractmethod
    def apply_parameter_defaults(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Apply default values to missing optional parameters.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model
            
        Returns:
            Configuration with defaults applied
        """
        pass


class BSRoformerValidatorInterface(ABC):
    """Specialized validator for BSRoformer models."""
    
    @abstractmethod
    def validate_freqs_per_bands(self, freqs_per_bands: Tuple[int, ...], stft_config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate frequency bands configuration.
        
        Args:
            freqs_per_bands: Tuple of frequencies per band
            stft_config: STFT configuration parameters
            
        Returns:
            List of validation issues for frequency bands
        """
        pass
    
    @abstractmethod
    def calculate_expected_freqs(self, stft_n_fft: int) -> int:
        """
        Calculate expected number of frequency bins from STFT configuration.
        
        Args:
            stft_n_fft: STFT n_fft parameter
            
        Returns:
            Expected number of frequency bins
        """
        pass


class MelBandRoformerValidatorInterface(ABC):
    """Specialized validator for MelBandRoformer models."""
    
    @abstractmethod
    def validate_num_bands(self, num_bands: int, sample_rate: int) -> List[ValidationIssue]:
        """
        Validate number of mel bands.
        
        Args:
            num_bands: Number of mel-scale bands
            sample_rate: Audio sample rate
            
        Returns:
            List of validation issues for mel bands configuration
        """
        pass
    
    @abstractmethod
    def validate_sample_rate(self, sample_rate: int) -> List[ValidationIssue]:
        """
        Validate audio sample rate.
        
        Args:
            sample_rate: Audio sample rate in Hz
            
        Returns:
            List of validation issues for sample rate
        """
        pass


class ConfigurationNormalizerInterface(ABC):
    """Interface for normalizing configuration between old and new formats."""
    
    @abstractmethod
    def normalize_config_format(self, raw_config: Any) -> Dict[str, Any]:
        """
        Normalize configuration from various input formats to standard dictionary.
        
        Args:
            raw_config: Configuration in any format (dict, object, etc.)
            
        Returns:
            Normalized configuration dictionary
        """
        pass
    
    @abstractmethod
    def map_legacy_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map legacy parameter names to current parameter names.
        
        Args:
            config: Configuration with potentially legacy parameter names
            
        Returns:
            Configuration with current parameter names
        """
        pass
    
    @abstractmethod
    def extract_nested_config(self, config: Any, path: str) -> Any:
        """
        Extract nested configuration value using dot notation path.
        
        Args:
            config: Configuration object or dictionary
            path: Dot notation path (e.g., "model.norm")
            
        Returns:
            Extracted configuration value or None if not found
        """
        pass
