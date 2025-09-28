"""
MelBandRoformer-specific configuration class.
Extends ModelConfiguration with MelBandRoformer-specific parameters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .model_configuration import ModelConfiguration, RoformerType


@dataclass(frozen=True, unsafe_hash=True)
class MelBandRoformerConfig(ModelConfiguration):
    """
    Configuration class specifically for MelBandRoformer models.
    
    MelBandRoformer processes audio using mel-scale frequency bands,
    which are more aligned with human auditory perception.
    """
    
    # MelBandRoformer-specific required parameters
    num_bands: int = 64  # Number of mel-scale bands
    
    # MelBandRoformer-specific optional parameters
    mel_scale: str = "htk"  # Mel scale type: "htk" or "slaney"
    fmin: float = 0.0  # Minimum frequency for mel scale
    fmax: Optional[float] = None  # Maximum frequency for mel scale (None = sample_rate/2)
    
    def __post_init__(self):
        """Validate MelBandRoformer-specific configuration after initialization."""
        super().__post_init__()
        self._validate_mel_band_roformer_parameters()
    
    def _validate_mel_band_roformer_parameters(self):
        """Validate MelBandRoformer-specific parameters."""
        if not isinstance(self.num_bands, int) or self.num_bands <= 0:
            raise ValueError(f"num_bands must be a positive integer, got {self.num_bands}")
        
        if self.num_bands < 8:
            raise ValueError(f"num_bands should be at least 8 for meaningful separation, got {self.num_bands}")
        
        if self.num_bands > 512:
            raise ValueError(f"num_bands should not exceed 512 for practical purposes, got {self.num_bands}")
        
        if self.mel_scale not in ["htk", "slaney"]:
            raise ValueError(f"mel_scale must be 'htk' or 'slaney', got '{self.mel_scale}'")
        
        if self.fmin < 0:
            raise ValueError(f"fmin must be non-negative, got {self.fmin}")
        
        if self.fmax is not None:
            if self.fmax <= self.fmin:
                raise ValueError(f"fmax ({self.fmax}) must be greater than fmin ({self.fmin})")
            
            if self.fmax > self.sample_rate / 2:
                raise ValueError(f"fmax ({self.fmax}) cannot exceed sample_rate/2 ({self.sample_rate/2})")
    
    def get_effective_fmax(self) -> float:
        """Get the effective maximum frequency (fmax or sample_rate/2)."""
        return self.fmax if self.fmax is not None else self.sample_rate / 2.0
    
    def get_frequency_range(self) -> tuple[float, float]:
        """Get the frequency range (fmin, fmax) for mel scale."""
        return (self.fmin, self.get_effective_fmax())
    
    def validate_sample_rate(self, sample_rate: int) -> bool:
        """
        Validate that the configuration is compatible with a given sample rate.
        
        Args:
            sample_rate: Audio sample rate to validate against
            
        Returns:
            True if compatible, False otherwise
        """
        if sample_rate != self.sample_rate:
            # Check if fmax is compatible with new sample rate
            if self.fmax is not None and self.fmax > sample_rate / 2:
                return False
        
        return True
    
    def get_mel_scale_info(self) -> Dict[str, Any]:
        """
        Get information about the mel scale configuration.
        
        Returns:
            Dictionary with mel scale information
        """
        return {
            'num_bands': self.num_bands,
            'mel_scale': self.mel_scale,
            'fmin': self.fmin,
            'fmax': self.get_effective_fmax(),
            'sample_rate': self.sample_rate,
            'frequency_range': self.get_frequency_range(),
            'bands_per_octave': self._estimate_bands_per_octave()
        }
    
    def _estimate_bands_per_octave(self) -> float:
        """Estimate the number of mel bands per octave."""
        fmin, fmax = self.get_frequency_range()
        if fmin <= 0:
            fmin = 1.0  # Avoid log(0)
        
        # Rough estimation using logarithmic scale
        import math
        octave_range = math.log2(fmax / fmin)
        return self.num_bands / octave_range if octave_range > 0 else self.num_bands
    
    def get_model_type(self) -> RoformerType:
        """Get the model type."""
        return RoformerType.MEL_BAND_ROFORMER
    
    def get_mel_band_roformer_kwargs(self) -> Dict[str, Any]:
        """Get MelBandRoformer-specific parameters for model initialization."""
        kwargs = self.get_transformer_kwargs()
        kwargs.update({
            'num_bands': self.num_bands,
            'sample_rate': self.sample_rate,
            'mel_scale': self.mel_scale,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'stereo': self.stereo,
            'num_stems': self.num_stems,
            'mlp_expansion_factor': self.mlp_expansion_factor,
            'use_torch_checkpoint': self.use_torch_checkpoint,
            'skip_connection': self.skip_connection,
        })
        return kwargs
    
    @classmethod
    def from_model_config(cls, config_dict: Dict[str, Any]) -> 'MelBandRoformerConfig':
        """
        Create MelBandRoformerConfig from a model configuration dictionary.
        
        Args:
            config_dict: Dictionary containing model configuration
            
        Returns:
            MelBandRoformerConfig instance
        """
        # Ensure num_bands is present
        if 'num_bands' not in config_dict:
            # Use default if not specified
            config_dict = config_dict.copy()
            config_dict['num_bands'] = 64
        
        return cls.from_dict(config_dict)
    
    def suggest_optimal_bands(self, target_frequency_resolution: float = 50.0) -> int:
        """
        Suggest optimal number of bands based on desired frequency resolution.
        
        Args:
            target_frequency_resolution: Desired frequency resolution in Hz
            
        Returns:
            Suggested number of bands
        """
        fmin, fmax = self.get_frequency_range()
        frequency_span = fmax - fmin
        suggested_bands = int(frequency_span / target_frequency_resolution)
        
        # Clamp to reasonable range
        return max(8, min(512, suggested_bands))
    
    def get_parameter_summary(self) -> str:
        """Get a summary string of key MelBandRoformer parameters."""
        base_summary = super().get_parameter_summary()
        mel_info = f", num_bands={self.num_bands}, sr={self.sample_rate}, fmax={self.get_effective_fmax():.0f}"
        return base_summary.replace(")", mel_info + ")")
    
    def __repr__(self) -> str:
        """String representation of the MelBandRoformer configuration."""
        return f"MelBandRoformerConfig({self.get_parameter_summary()})"
