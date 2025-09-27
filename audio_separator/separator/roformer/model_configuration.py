"""
Model configuration dataclass for Roformer models.
Supports both old and new parameter sets with backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
from enum import Enum


class RoformerType(Enum):
    """Supported Roformer model types."""
    BS_ROFORMER = "bs_roformer"
    MEL_BAND_ROFORMER = "mel_band_roformer"


@dataclass(frozen=True, unsafe_hash=True)
class ModelConfiguration:
    """
    Model configuration parameters for Roformer models.
    
    This class supports both old and new parameter sets to maintain
    backward compatibility while enabling new features.
    """
    
    # Required parameters (must be provided)
    dim: int
    depth: int
    
    # Common optional parameters (with sensible defaults)
    stereo: bool = False
    num_stems: int = 1
    time_transformer_depth: int = 2
    freq_transformer_depth: int = 2
    dim_head: int = 64
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    flash_attn: bool = True
    
    # New parameters (with defaults for backward compatibility)
    mlp_expansion_factor: int = 4
    sage_attention: bool = False
    zero_dc: bool = True
    use_torch_checkpoint: bool = False
    skip_connection: bool = False
    
    # Normalization (may be None in some configs)
    norm: Optional[str] = None
    
    # Model-specific parameters (set by subclasses)
    freqs_per_bands: Optional[Tuple[int, ...]] = None  # BSRoformer
    num_bands: Optional[int] = None  # MelBandRoformer
    sample_rate: int = 44100  # Default sample rate
    
    # Additional configuration data (stored as tuple for hashability)
    extra_config: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_basic_parameters()
        self._validate_parameter_ranges()
    
    def _validate_basic_parameters(self):
        """Validate basic parameter types and requirements."""
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {self.dim}")
        
        if not isinstance(self.depth, int) or self.depth <= 0:
            raise ValueError(f"depth must be a positive integer, got {self.depth}")
        
        if not isinstance(self.num_stems, int) or self.num_stems <= 0:
            raise ValueError(f"num_stems must be a positive integer, got {self.num_stems}")
        
        if not isinstance(self.mlp_expansion_factor, int) or self.mlp_expansion_factor <= 0:
            raise ValueError(f"mlp_expansion_factor must be a positive integer, got {self.mlp_expansion_factor}")
    
    def _validate_parameter_ranges(self):
        """Validate parameter value ranges."""
        if not (0.0 <= self.attn_dropout <= 1.0):
            raise ValueError(f"attn_dropout must be between 0.0 and 1.0, got {self.attn_dropout}")
        
        if not (0.0 <= self.ff_dropout <= 1.0):
            raise ValueError(f"ff_dropout must be between 0.0 and 1.0, got {self.ff_dropout}")
        
        if self.dim_head <= 0:
            raise ValueError(f"dim_head must be positive, got {self.dim_head}")
        
        if self.heads <= 0:
            raise ValueError(f"heads must be positive, got {self.heads}")
        
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            if value is not None:
                result[field_info.name] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfiguration':
        """Create configuration from dictionary."""
        # Extract known parameters
        known_params = {}
        extra_params = {}
        
        for key, value in config_dict.items():
            if key in cls.__dataclass_fields__:
                known_params[key] = value
            else:
                extra_params[key] = value
        
        # Set extra_config as tuple if there are unknown parameters
        if extra_params:
            known_params['extra_config'] = tuple(extra_params.items())
        
        return cls(**known_params)
    
    def get_transformer_kwargs(self) -> Dict[str, Any]:
        """Get parameters to pass to transformer initialization."""
        return {
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'attn_dropout': self.attn_dropout,
            'ff_dropout': self.ff_dropout,
            'flash_attn': self.flash_attn,
            'sage_attention': self.sage_attention,  # New parameter
            'zero_dc': self.zero_dc,  # New parameter
        }
    
    def has_new_parameters(self) -> bool:
        """Check if configuration uses any new parameters."""
        return (
            self.mlp_expansion_factor != 4 or
            self.sage_attention is True or
            self.zero_dc is not True or
            self.use_torch_checkpoint is True or
            self.skip_connection is True
        )
    
    def get_parameter_summary(self) -> str:
        """Get a summary string of key parameters."""
        return (
            f"ModelConfiguration(dim={self.dim}, depth={self.depth}, "
            f"stems={self.num_stems}, mlp_factor={self.mlp_expansion_factor}, "
            f"sage_attn={self.sage_attention}, new_params={self.has_new_parameters()})"
        )
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return self.get_parameter_summary()
