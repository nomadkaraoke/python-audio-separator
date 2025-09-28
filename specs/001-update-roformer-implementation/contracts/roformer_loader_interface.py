"""
API Contract: Roformer Model Loader Interface
This defines the interface for loading Roformer models with backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class RoformerType(Enum):
    """Supported Roformer model types."""
    BS_ROFORMER = "bs_roformer"
    MEL_BAND_ROFORMER = "mel_band_roformer"


class ImplementationVersion(Enum):
    """Available implementation versions."""
    OLD = "old"
    NEW = "new"
    FALLBACK = "fallback"


@dataclass
class ModelLoadingResult:
    """Result of model loading operation."""
    success: bool
    model: Optional[Any] = None  # Actual model instance
    error_message: Optional[str] = None
    implementation_used: ImplementationVersion = ImplementationVersion.NEW
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ModelConfiguration:
    """Model configuration parameters."""
    # Required parameters
    dim: int
    depth: int
    
    # Common optional parameters
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
    
    # Model-specific parameters
    freqs_per_bands: Optional[Tuple[int, ...]] = None  # BSRoformer
    num_bands: Optional[int] = None  # MelBandRoformer


class ParameterValidationError(Exception):
    """Raised when model parameters are invalid."""
    
    def __init__(self, parameter_name: str, expected_type: str, actual_value: Any, suggested_fix: str):
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.suggested_fix = suggested_fix
        
        message = (
            f"Invalid parameter '{parameter_name}': "
            f"expected {expected_type}, got {type(actual_value).__name__} ({actual_value}). "
            f"Suggestion: {suggested_fix}"
        )
        super().__init__(message)


class RoformerLoaderInterface(ABC):
    """Abstract interface for loading Roformer models."""
    
    @abstractmethod
    def load_model(self, model_path: str, config: Optional[Dict[str, Any]] = None) -> ModelLoadingResult:
        """
        Load a Roformer model from the given path.
        
        Args:
            model_path: Path to the model file (.ckpt, .pth)
            config: Optional configuration override
            
        Returns:
            ModelLoadingResult with success status and model or error details
            
        Raises:
            ParameterValidationError: If model configuration is invalid
            FileNotFoundError: If model file doesn't exist
        """
        pass
    
    @abstractmethod
    def validate_configuration(self, config: ModelConfiguration, model_type: RoformerType) -> List[str]:
        """
        Validate model configuration parameters.
        
        Args:
            config: Model configuration to validate
            model_type: Type of Roformer model
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def detect_model_type(self, model_path: str) -> RoformerType:
        """
        Detect the type of Roformer model from the file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Detected RoformerType
            
        Raises:
            ValueError: If model type cannot be determined
        """
        pass
    
    @abstractmethod
    def get_default_configuration(self, model_type: RoformerType) -> ModelConfiguration:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of Roformer model
            
        Returns:
            Default ModelConfiguration for the type
        """
        pass


class RoformerModelInterface(ABC):
    """Abstract interface for Roformer model instances."""
    
    @abstractmethod
    def separate_audio(self, audio_data: Any, **kwargs) -> Any:
        """
        Separate audio into stems using the model.
        
        Args:
            audio_data: Input audio data
            **kwargs: Additional separation parameters
            
        Returns:
            Separated audio stems
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass


class FallbackLoaderInterface(ABC):
    """Interface for fallback loading mechanism."""
    
    @abstractmethod
    def try_new_implementation(self, model_path: str, config: ModelConfiguration) -> ModelLoadingResult:
        """
        Attempt to load model with new implementation.
        
        Args:
            model_path: Path to model file
            config: Model configuration
            
        Returns:
            ModelLoadingResult indicating success or failure
        """
        pass
    
    @abstractmethod
    def try_old_implementation(self, model_path: str, config: ModelConfiguration) -> ModelLoadingResult:
        """
        Attempt to load model with old implementation (fallback).
        
        Args:
            model_path: Path to model file
            config: Model configuration
            
        Returns:
            ModelLoadingResult indicating success or failure
        """
        pass
    
    @abstractmethod
    def should_fallback(self, error: Exception) -> bool:
        """
        Determine if the error warrants falling back to old implementation.
        
        Args:
            error: Exception from new implementation attempt
            
        Returns:
            True if should attempt fallback, False otherwise
        """
        pass
