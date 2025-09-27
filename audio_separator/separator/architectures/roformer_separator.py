"""
Roformer-specific separator implementation.
Handles Roformer model loading and separation with new parameter support.
"""

from typing import Dict, Any, Optional
import logging
import numpy as np

from ..common_separator import CommonSeparator
from ..roformer.roformer_loader import RoformerLoader
from ..roformer.model_loading_result import ModelLoadingResult

logger = logging.getLogger(__name__)


class RoformerSeparator(CommonSeparator):
    """
    Roformer-specific separator using the new loading system.
    
    This separator provides dedicated handling for Roformer models
    with the new parameter validation and fallback mechanisms.
    """
    
    def __init__(self, common_config, arch_specific_params):
        """
        Initialize the Roformer separator.
        
        Args:
            common_config: Common separator configuration
            arch_specific_params: Roformer-specific parameters
        """
        super().__init__(common_config, arch_specific_params)
        
        self.roformer_loader = RoformerLoader()
        self.model_loading_result: Optional[ModelLoadingResult] = None
        
        # Roformer-specific configuration
        self.is_roformer = True
        self.model_type = None  # Will be detected during loading
        
        logger.info("Initialized RoformerSeparator with new loading system")
    
    def load_model(self):
        """
        Load the Roformer model using the new loading system.
        
        This method uses the RoformerLoader with fallback mechanisms
        to handle both new and legacy Roformer models.
        """
        try:
            # Use the new loading system
            self.model_loading_result = self.roformer_loader.load_model(
                model_path=self.model_file_path,
                config=self.model_data_cfgdict.model,
                device=str(self.torch_device)
            )
            
            if self.model_loading_result.success:
                self.model = self.model_loading_result.model
                self.model_type = self.model_loading_result.model_type
                
                logger.info(f"Successfully loaded {self.model_type} model")
                logger.info(f"Implementation: {self.model_loading_result.implementation_version}")
                logger.info(f"Loading method: {self.model_loading_result.loading_method}")
                
            else:
                error_msg = f"Failed to load Roformer model: {self.model_loading_result.error_message}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"Error loading Roformer model: {e}")
            raise
    
    def demix(self, mix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform source separation on the input mix.
        
        Args:
            mix: Input audio mixture as numpy array
            
        Returns:
            Dictionary mapping stem names to separated audio arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Demixing audio with shape: {mix.shape}")
        
        # Use the model's forward method for separation
        # This is a simplified implementation - in practice, you'd need
        # to handle chunking, overlap-add, etc. based on the model type
        
        try:
            # Convert to tensor and add batch dimension
            import torch
            
            if len(mix.shape) == 1:
                mix = mix.reshape(1, -1)  # Add channel dimension
            
            mix_tensor = torch.from_numpy(mix).float().to(self.torch_device)
            
            if len(mix_tensor.shape) == 2:
                mix_tensor = mix_tensor.unsqueeze(0)  # Add batch dimension
            
            # Perform separation
            with torch.no_grad():
                separated = self.model(mix_tensor)
            
            # Convert back to numpy
            if isinstance(separated, torch.Tensor):
                separated = separated.cpu().numpy()
            
            # Handle different output formats
            result = {}
            if len(separated.shape) == 3:  # [batch, stems, samples]
                for i in range(separated.shape[1]):
                    stem_name = f"stem_{i}"
                    result[stem_name] = separated[0, i, :]
            else:
                result["separated"] = separated[0] if len(separated.shape) > 1 else separated
            
            logger.debug(f"Separation complete. Output stems: {list(result.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"Error during demixing: {e}")
            raise
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model loading process.
        
        Returns:
            Dictionary with loading statistics and model information
        """
        stats = self.roformer_loader.get_loading_stats()
        
        if self.model_loading_result:
            stats.update({
                'current_model_type': self.model_loading_result.model_type,
                'implementation_version': self.model_loading_result.implementation_version,
                'loading_method': self.model_loading_result.loading_method,
                'device': self.model_loading_result.device,
            })
        
        return stats
    
    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a model configuration before loading.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if self.model_type:
            return self.roformer_loader.validate_configuration(config, self.model_type)
        else:
            # Try both model types if type is unknown
            return (self.roformer_loader.validate_configuration(config, "bs_roformer") or
                   self.roformer_loader.validate_configuration(config, "mel_band_roformer"))
