"""
Fallback loader for legacy Roformer models.
Provides compatibility with older model implementations.
"""

from typing import Dict, Any, Optional, Union
import logging
import sys
import os

# Add contracts to path for interface imports (optional)
try:
    sys.path.append('/Users/andrew/Projects/python-audio-separator/specs/001-update-roformer-implementation/contracts')
    from fallback_loader_interface import FallbackLoaderInterface, ModelLoadingResult
    _has_interface = True
except ImportError:
    # Create a dummy interface for when contracts are not available
    class FallbackLoaderInterface:
        pass
    # Use local ModelLoadingResult if contract not available
    from dataclasses import dataclass
    from typing import Any, Dict
    
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
    _has_interface = False

logger = logging.getLogger(__name__)


class FallbackLoader(FallbackLoaderInterface):
    """
    Fallback loader for legacy Roformer model implementations.
    
    This loader attempts to load models using the old parameter sets
    and implementation approaches when the new implementation fails.
    """
    
    def __init__(self):
        """Initialize the fallback loader."""
        self._fallback_attempts = 0
        self._fallback_successes = 0
    
    def try_new_implementation(self, 
                             model_path: str, 
                             config: Dict[str, Any], 
                             device: str = 'cpu') -> ModelLoadingResult:
        """
        Try loading with new implementation (delegated to main loader).
        
        This method is part of the interface but in practice is handled
        by the main RoformerLoader. This implementation serves as a
        placeholder for interface compliance.
        
        Args:
            model_path: Path to the model file
            config: Model configuration
            device: Device to load on
            
        Returns:
            ModelLoadingResult indicating this should be handled by main loader
        """
        return ModelLoadingResult(
            model=None,
            model_type="unknown",
            config_used=config,
            implementation_version="new",
            loading_method="delegated",
            device=device,
            success=False,
            error_message="New implementation should be handled by RoformerLoader"
        )
    
    def try_legacy_implementation(self, 
                                model_path: str, 
                                config: Dict[str, Any], 
                                device: str = 'cpu') -> ModelLoadingResult:
        """
        Try loading with legacy implementation.
        
        This method attempts various fallback strategies:
        1. Load with minimal parameter set (ignore new parameters)
        2. Try different model instantiation approaches
        3. Handle different state dict formats
        
        Args:
            model_path: Path to the model file
            config: Model configuration
            device: Device to load on
            
        Returns:
            ModelLoadingResult with fallback attempt results
        """
        self._fallback_attempts += 1
        logger.info(f"Attempting fallback loading for {model_path}")
        
        # Try different fallback strategies in order
        strategies = [
            self._try_minimal_parameters,
            self._try_legacy_constructor,
            self._try_parameter_filtering,
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                logger.debug(f"Trying fallback strategy {i}/{len(strategies)}: {strategy.__name__}")
                result = strategy(model_path, config, device)
                
                if result.success:
                    self._fallback_successes += 1
                    logger.info(f"Fallback strategy {i} succeeded: {strategy.__name__}")
                    return result
                    
            except Exception as e:
                logger.debug(f"Fallback strategy {i} failed: {e}")
                continue
        
        # All strategies failed
        logger.error("All fallback strategies failed")
        return ModelLoadingResult(
            model=None,
            model_type="unknown",
            config_used=config,
            implementation_version="legacy",
            loading_method="fallback_failed",
            device=device,
            success=False,
            error_message="All fallback strategies exhausted"
        )
    
    def _try_minimal_parameters(self, 
                              model_path: str, 
                              config: Dict[str, Any], 
                              device: str) -> ModelLoadingResult:
        """
        Try loading with only the minimal required parameters.
        
        This strategy filters out new parameters that might not be
        supported by older model implementations.
        """
        # Define minimal parameter sets for each model type
        bs_roformer_minimal = {
            'dim', 'depth', 'freqs_per_bands', 'stereo', 'num_stems',
            'time_transformer_depth', 'freq_transformer_depth', 
            'dim_head', 'heads', 'attn_dropout', 'ff_dropout', 'flash_attn'
        }
        
        mel_roformer_minimal = {
            'dim', 'depth', 'num_bands', 'stereo', 'num_stems',
            'time_transformer_depth', 'freq_transformer_depth',
            'dim_head', 'heads', 'attn_dropout', 'ff_dropout', 'flash_attn'
        }
        
        # Detect model type and filter parameters
        if 'freqs_per_bands' in config:
            model_type = "bs_roformer"
            filtered_config = {k: v for k, v in config.items() 
                             if k in bs_roformer_minimal}
        elif 'num_bands' in config:
            model_type = "mel_band_roformer"
            filtered_config = {k: v for k, v in config.items() 
                             if k in mel_roformer_minimal}
        else:
            # Default to BSRoformer with basic parameters
            model_type = "bs_roformer"
            filtered_config = {k: v for k, v in config.items() 
                             if k in bs_roformer_minimal}
            # Add default freqs_per_bands if missing
            if 'freqs_per_bands' not in filtered_config:
                filtered_config['freqs_per_bands'] = (2, 4, 8, 16, 32, 64)
        
        logger.debug(f"Minimal parameters: {list(filtered_config.keys())}")
        
        try:
            model = self._create_model_with_config(model_type, filtered_config)
            model = self._load_state_dict_flexible(model, model_path, device)
            
            return ModelLoadingResult(
                model=model,
                model_type=model_type,
                config_used=filtered_config,
                implementation_version="legacy",
                loading_method="minimal_parameters",
                device=device,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.debug(f"Minimal parameters strategy failed: {e}")
            raise
    
    def _try_legacy_constructor(self, 
                              model_path: str, 
                              config: Dict[str, Any], 
                              device: str) -> ModelLoadingResult:
        """
        Try using legacy constructor patterns.
        
        This strategy attempts to use older ways of instantiating the models
        that might be more compatible with legacy saved states.
        """
        # Try to load with very basic parameters only
        basic_config = {
            'dim': config.get('dim', 512),
            'depth': config.get('depth', 12),
        }
        
        # Add model-specific basics
        if 'freqs_per_bands' in config:
            model_type = "bs_roformer"
            basic_config['freqs_per_bands'] = config['freqs_per_bands']
        elif 'num_bands' in config:
            model_type = "mel_band_roformer"
            basic_config['num_bands'] = config['num_bands']
        else:
            model_type = "bs_roformer"
            basic_config['freqs_per_bands'] = (2, 4, 8, 16, 32, 64)
        
        logger.debug(f"Legacy constructor with: {basic_config}")
        
        try:
            model = self._create_model_with_config(model_type, basic_config)
            model = self._load_state_dict_flexible(model, model_path, device)
            
            return ModelLoadingResult(
                model=model,
                model_type=model_type,
                config_used=basic_config,
                implementation_version="legacy",
                loading_method="legacy_constructor",
                device=device,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.debug(f"Legacy constructor strategy failed: {e}")
            raise
    
    def _try_parameter_filtering(self, 
                               model_path: str, 
                               config: Dict[str, Any], 
                               device: str) -> ModelLoadingResult:
        """
        Try filtering out problematic parameters one by one.
        
        This strategy systematically removes parameters that might be
        causing issues until it finds a working combination.
        """
        # Parameters that are known to cause issues in legacy models
        problematic_params = [
            'mlp_expansion_factor', 'sage_attention', 'zero_dc',
            'use_torch_checkpoint', 'skip_connection', 'norm', 'act'
        ]
        
        working_config = config.copy()
        
        # Remove problematic parameters
        for param in problematic_params:
            if param in working_config:
                del working_config[param]
                logger.debug(f"Removed problematic parameter: {param}")
        
        # Determine model type
        if 'freqs_per_bands' in working_config:
            model_type = "bs_roformer"
        elif 'num_bands' in working_config:
            model_type = "mel_band_roformer"
        else:
            model_type = "bs_roformer"
            working_config['freqs_per_bands'] = (2, 4, 8, 16, 32, 64)
        
        logger.debug(f"Filtered config: {list(working_config.keys())}")
        
        try:
            model = self._create_model_with_config(model_type, working_config)
            model = self._load_state_dict_flexible(model, model_path, device)
            
            return ModelLoadingResult(
                model=model,
                model_type=model_type,
                config_used=working_config,
                implementation_version="legacy",
                loading_method="parameter_filtering",
                device=device,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.debug(f"Parameter filtering strategy failed: {e}")
            raise
    
    def _create_model_with_config(self, model_type: str, config: Dict[str, Any]):
        """Create model instance with given configuration."""
        if model_type == "bs_roformer":
            from ..uvr_lib_v5.roformer.bs_roformer import BSRoformer
            return BSRoformer(**config)
        elif model_type == "mel_band_roformer":
            from ..uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
            return MelBandRoformer(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_state_dict_flexible(self, model, model_path: str, device: str):
        """Load state dict with flexible handling of different formats."""
        import torch
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file does not exist: {model_path}")
            return model
        
        try:
            # Try loading the state dict
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Try loading with strict=False to ignore missing/extra keys
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.debug("Loaded state dict with strict=True")
            except Exception as e:
                logger.debug(f"Strict loading failed: {e}, trying strict=False")
                model.load_state_dict(state_dict, strict=False)
                logger.debug("Loaded state dict with strict=False")
            
            model.to(device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load state dict: {e}")
            raise
    
    def get_fallback_stats(self) -> Dict[str, int]:
        """Get fallback loading statistics."""
        return {
            'attempts': self._fallback_attempts,
            'successes': self._fallback_successes,
            'success_rate': self._fallback_successes / max(self._fallback_attempts, 1)
        }
