"""
Interface contract for fallback loader implementations.
Defines the expected behavior for fallback loading mechanisms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelLoadingResult:
    """Result of a model loading attempt."""
    model: Any
    model_type: str
    config_used: Dict[str, Any]
    implementation_version: str
    loading_method: str
    device: str
    success: bool
    error_message: str = None


class FallbackLoaderInterface(ABC):
    """
    Interface for fallback model loading implementations.
    
    Defines the contract that fallback loaders must implement to provide
    compatibility with legacy models when new implementations fail.
    """
    
    @abstractmethod
    def try_new_implementation(self, 
                             model_path: str, 
                             config: Dict[str, Any], 
                             device: str = 'cpu') -> ModelLoadingResult:
        """
        Try loading with new implementation.
        
        Args:
            model_path: Path to the model file
            config: Model configuration dictionary
            device: Device to load model on
            
        Returns:
            ModelLoadingResult with attempt results
        """
        pass
    
    @abstractmethod
    def try_legacy_implementation(self, 
                                model_path: str, 
                                config: Dict[str, Any], 
                                device: str = 'cpu') -> ModelLoadingResult:
        """
        Try loading with legacy implementation.
        
        Args:
            model_path: Path to the model file
            config: Model configuration dictionary
            device: Device to load model on
            
        Returns:
            ModelLoadingResult with fallback attempt results
        """
        pass
