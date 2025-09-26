"""
BSRoformer-specific configuration class.
Extends ModelConfiguration with BSRoformer-specific parameters.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from .model_configuration import ModelConfiguration, RoformerType


@dataclass(frozen=True, unsafe_hash=True)
class BSRoformerConfig(ModelConfiguration):
    """
    Configuration class specifically for BSRoformer (Band-Split Roformer) models.
    
    BSRoformer processes audio by splitting it into frequency bands and applying
    Roformer architecture to each band separately.
    """
    
    # BSRoformer-specific required parameters
    freqs_per_bands: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)  # Default frequency band configuration
    
    # BSRoformer-specific optional parameters
    mask_estimator_depth: int = 2  # Depth of mask estimation network
    
    def __post_init__(self):
        """Validate BSRoformer-specific configuration after initialization."""
        super().__post_init__()
        self._validate_bs_roformer_parameters()
    
    def _validate_bs_roformer_parameters(self):
        """Validate BSRoformer-specific parameters."""
        if not self.freqs_per_bands:
            raise ValueError("freqs_per_bands must be provided for BSRoformer")
        
        if not isinstance(self.freqs_per_bands, (tuple, list)):
            raise ValueError(f"freqs_per_bands must be a tuple or list, got {type(self.freqs_per_bands)}")
        
        if not all(isinstance(freq, int) and freq > 0 for freq in self.freqs_per_bands):
            raise ValueError("All frequencies in freqs_per_bands must be positive integers")
        
        if len(self.freqs_per_bands) < 2:
            raise ValueError("freqs_per_bands must contain at least 2 frequency bands")
        
        if self.mask_estimator_depth <= 0:
            raise ValueError(f"mask_estimator_depth must be positive, got {self.mask_estimator_depth}")
    
    def get_total_frequency_bins(self) -> int:
        """Calculate total number of frequency bins."""
        return sum(self.freqs_per_bands)
    
    def validate_against_stft_config(self, n_fft: int) -> bool:
        """
        Validate that frequency bands match STFT configuration.
        
        Args:
            n_fft: STFT n_fft parameter
            
        Returns:
            True if configuration is valid, False otherwise
        """
        expected_freq_bins = n_fft // 2 + 1
        total_freq_bins = self.get_total_frequency_bins()
        
        return total_freq_bins == expected_freq_bins
    
    def get_stft_compatibility_info(self, n_fft: int) -> Dict[str, Any]:
        """
        Get information about STFT compatibility.
        
        Args:
            n_fft: STFT n_fft parameter
            
        Returns:
            Dictionary with compatibility information
        """
        expected_freq_bins = n_fft // 2 + 1
        total_freq_bins = self.get_total_frequency_bins()
        
        return {
            'expected_freq_bins': expected_freq_bins,
            'actual_freq_bins': total_freq_bins,
            'is_compatible': total_freq_bins == expected_freq_bins,
            'difference': total_freq_bins - expected_freq_bins,
            'freqs_per_bands': self.freqs_per_bands
        }
    
    def get_model_type(self) -> RoformerType:
        """Get the model type."""
        return RoformerType.BS_ROFORMER
    
    def get_bs_roformer_kwargs(self) -> Dict[str, Any]:
        """Get BSRoformer-specific parameters for model initialization."""
        kwargs = self.get_transformer_kwargs()
        kwargs.update({
            'freqs_per_bands': self.freqs_per_bands,
            'mask_estimator_depth': self.mask_estimator_depth,
            'stereo': self.stereo,
            'num_stems': self.num_stems,
            'mlp_expansion_factor': self.mlp_expansion_factor,
            'use_torch_checkpoint': self.use_torch_checkpoint,
            'skip_connection': self.skip_connection,
        })
        return kwargs
    
    @classmethod
    def from_model_config(cls, config_dict: Dict[str, Any]) -> 'BSRoformerConfig':
        """
        Create BSRoformerConfig from a model configuration dictionary.
        
        Args:
            config_dict: Dictionary containing model configuration
            
        Returns:
            BSRoformerConfig instance
        """
        # Ensure freqs_per_bands is present
        if 'freqs_per_bands' not in config_dict:
            # Use default if not specified
            config_dict = config_dict.copy()
            config_dict['freqs_per_bands'] = (2, 4, 8, 16, 32, 64)
        
        return cls.from_dict(config_dict)
    
    def suggest_stft_n_fft(self) -> int:
        """
        Suggest appropriate n_fft value for STFT based on frequency bands.
        
        Returns:
            Suggested n_fft value
        """
        total_freq_bins = self.get_total_frequency_bins()
        # n_fft = 2 * (freq_bins - 1)
        return 2 * (total_freq_bins - 1)
    
    def get_parameter_summary(self) -> str:
        """Get a summary string of key BSRoformer parameters."""
        base_summary = super().get_parameter_summary()
        bs_info = f", freqs_per_bands={self.freqs_per_bands}, total_bins={self.get_total_frequency_bins()}"
        return base_summary.replace(")", bs_info + ")")
    
    def __repr__(self) -> str:
        """String representation of the BSRoformer configuration."""
        return f"BSRoformerConfig({self.get_parameter_summary()})"
