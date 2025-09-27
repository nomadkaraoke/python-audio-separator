"""
Model loading result dataclass.
Contains the result of model loading operations with success/failure information.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from enum import Enum


class ImplementationVersion(Enum):
    """Available implementation versions."""
    OLD = "old"
    NEW = "new"
    FALLBACK = "fallback"


@dataclass
class ModelLoadingResult:
    """
    Result of a model loading operation.
    
    Contains information about whether the loading was successful,
    the loaded model (if successful), error details (if failed),
    and metadata about which implementation was used.
    """
    
    success: bool
    model: Optional[Any] = None  # Actual model instance (torch.nn.Module)
    error_message: Optional[str] = None
    implementation_used: ImplementationVersion = ImplementationVersion.NEW
    warnings: List[str] = field(default_factory=list)
    
    # Additional metadata
    loading_time_seconds: Optional[float] = None
    model_info: Dict[str, Any] = field(default_factory=dict)
    config_used: Optional[Any] = None  # The configuration that was actually used
    
    def __post_init__(self):
        """Validate result after initialization."""
        if self.success and self.model is None:
            raise ValueError("success=True but model is None")
        
        if not self.success and self.error_message is None:
            raise ValueError("success=False but error_message is None")
        
        if self.warnings is None:
            self.warnings = []
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def add_model_info(self, key: str, value: Any):
        """Add model metadata information."""
        self.model_info[key] = value
    
    def get_summary(self) -> str:
        """Get a summary string of the loading result."""
        if self.success:
            summary = f"SUCCESS: Model loaded using {self.implementation_used.value} implementation"
            if self.loading_time_seconds:
                summary += f" in {self.loading_time_seconds:.2f}s"
            if self.warnings:
                summary += f" with {len(self.warnings)} warnings"
        else:
            summary = f"FAILED: {self.error_message}"
        
        return summary
    
    def is_fallback_used(self) -> bool:
        """Check if fallback implementation was used."""
        return self.implementation_used == ImplementationVersion.FALLBACK
    
    def is_new_implementation_used(self) -> bool:
        """Check if new implementation was used."""
        return self.implementation_used == ImplementationVersion.NEW
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_model_parameters_count(self) -> Optional[int]:
        """Get the number of model parameters if available."""
        if self.model is None:
            return None
        
        try:
            # Try to count parameters for PyTorch models
            if hasattr(self.model, 'parameters'):
                return sum(p.numel() for p in self.model.parameters())
        except Exception:
            pass
        
        return self.model_info.get('parameter_count')
    
    def get_model_size_mb(self) -> Optional[float]:
        """Get the model size in MB if available."""
        param_count = self.get_model_parameters_count()
        if param_count is not None:
            # Assume float32 parameters (4 bytes each)
            return (param_count * 4) / (1024 * 1024)
        
        return self.model_info.get('size_mb')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'implementation_used': self.implementation_used.value,
            'warnings': self.warnings,
            'loading_time_seconds': self.loading_time_seconds,
            'model_info': self.model_info,
            'has_model': self.model is not None,
            'parameter_count': self.get_model_parameters_count(),
            'model_size_mb': self.get_model_size_mb(),
        }
    
    @classmethod
    def success_result(
        cls, 
        model: Any, 
        implementation: ImplementationVersion = ImplementationVersion.NEW,
        config: Optional[Any] = None,
        loading_time: Optional[float] = None
    ) -> 'ModelLoadingResult':
        """
        Create a successful loading result.
        
        Args:
            model: The loaded model instance
            implementation: Which implementation was used
            config: The configuration that was used
            loading_time: Time taken to load the model
            
        Returns:
            ModelLoadingResult indicating success
        """
        return cls(
            success=True,
            model=model,
            implementation_used=implementation,
            config_used=config,
            loading_time_seconds=loading_time
        )
    
    @classmethod
    def failure_result(
        cls, 
        error_message: str,
        implementation: ImplementationVersion = ImplementationVersion.NEW,
        warnings: Optional[List[str]] = None
    ) -> 'ModelLoadingResult':
        """
        Create a failed loading result.
        
        Args:
            error_message: Description of what went wrong
            implementation: Which implementation was attempted
            warnings: Any warnings that occurred before failure
            
        Returns:
            ModelLoadingResult indicating failure
        """
        return cls(
            success=False,
            error_message=error_message,
            implementation_used=implementation,
            warnings=warnings or []
        )
    
    @classmethod
    def fallback_success_result(
        cls,
        model: Any,
        original_error: str,
        config: Optional[Any] = None,
        loading_time: Optional[float] = None
    ) -> 'ModelLoadingResult':
        """
        Create a successful fallback loading result.
        
        Args:
            model: The loaded model instance
            original_error: The error that caused fallback
            config: The configuration that was used
            loading_time: Time taken to load the model
            
        Returns:
            ModelLoadingResult indicating fallback success
        """
        result = cls(
            success=True,
            model=model,
            implementation_used=ImplementationVersion.FALLBACK,
            config_used=config,
            loading_time_seconds=loading_time
        )
        result.add_warning(f"Fell back to old implementation due to: {original_error}")
        return result
    
    def __str__(self) -> str:
        """String representation of the loading result."""
        return self.get_summary()
    
    def __repr__(self) -> str:
        """Detailed string representation of the loading result."""
        return f"ModelLoadingResult(success={self.success}, impl={self.implementation_used.value}, warnings={len(self.warnings)})"
