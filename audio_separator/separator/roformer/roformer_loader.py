"""
Roformer model loader with fallback mechanism.
Handles loading of both new and legacy Roformer models.
"""

from typing import Dict, Any, Optional, Tuple, Union
import logging
import sys
import os

# Add contracts to path for interface imports (optional)
try:
    # Find project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    # Go up until we find the project root (contains specs/ directory)
    while project_root and not os.path.exists(os.path.join(project_root, 'specs')):
        parent = os.path.dirname(project_root)
        if parent == project_root:  # Reached filesystem root
            break
        project_root = parent
    
    contracts_path = os.path.join(project_root, 'specs', '001-update-roformer-implementation', 'contracts')
    if os.path.exists(contracts_path):
        sys.path.append(contracts_path)
    from roformer_loader_interface import RoformerLoaderInterface
    _has_interface = True
except ImportError:
    # Create a dummy interface for when contracts are not available
    class RoformerLoaderInterface:
        pass
    _has_interface = False
from .model_loading_result import ModelLoadingResult
from .configuration_normalizer import ConfigurationNormalizer
from .parameter_validation_error import ParameterValidationError
from .fallback_loader import FallbackLoader

logger = logging.getLogger(__name__)


class RoformerLoader(RoformerLoaderInterface):
    """
    Main Roformer model loader with fallback mechanism.
    
    Attempts to load models with new parameter handling first,
    then falls back to legacy loading if that fails.
    """
    
    def __init__(self):
        """Initialize the loader with normalizer and fallback loader."""
        self.config_normalizer = ConfigurationNormalizer()
        self.fallback_loader = FallbackLoader()
        self._loading_stats = {
            'new_implementation_success': 0,
            'fallback_success': 0,
            'total_failures': 0
        }
    
    def load_model(self, 
                  model_path: str, 
                  config: Dict[str, Any], 
                  device: str = 'cpu') -> ModelLoadingResult:
        """
        Load a Roformer model with fallback mechanism.
        
        Args:
            model_path: Path to the model file
            config: Model configuration dictionary
            device: Device to load model on
            
        Returns:
            ModelLoadingResult containing the loaded model and metadata
        """
        logger.info(f"Loading Roformer model from {model_path}")
        
        # Step 1: Normalize and validate configuration
        try:
            normalized_config = self.config_normalizer.normalize_from_file_path(
                config, model_path, apply_defaults=True, validate=True
            )
            model_type = self.config_normalizer.detect_model_type(normalized_config)
            
            logger.debug(f"Detected model type: {model_type}")
            logger.debug(f"Normalized config: {len(normalized_config)} parameters")
            
        except ParameterValidationError as e:
            logger.warning(f"Configuration validation failed: {e}")
            # Try fallback with original config
            return self._try_fallback_loading(model_path, config, device, 
                                            failure_reason=f"Config validation: {e}")
        
        # Step 2: Try new implementation first
        try:
            result = self._load_with_new_implementation(
                model_path, normalized_config, model_type, device
            )
            
            self._loading_stats['new_implementation_success'] += 1
            logger.info(f"Successfully loaded {model_type} model with new implementation")
            
            return result
            
        except Exception as e:
            logger.warning(f"New implementation failed: {e}")
            logger.debug("Falling back to legacy implementation")
            
            # Step 3: Try fallback implementation
            return self._try_fallback_loading(model_path, config, device,
                                            failure_reason=f"New implementation: {e}")
    
    def validate_configuration(self, config: Dict[str, Any], model_type: str) -> bool:
        """
        Validate a configuration dictionary for the specified model type.
        
        Args:
            config: Configuration dictionary to validate
            model_type: Type of model ("bs_roformer" or "mel_band_roformer")
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            normalized_config = self.config_normalizer.normalize_config(
                config, model_type, apply_defaults=False, validate=True
            )
            logger.debug(f"Configuration validation passed for {model_type}")
            return True
            
        except ParameterValidationError as e:
            logger.warning(f"Configuration validation failed for {model_type}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False
    
    def _load_with_new_implementation(self, 
                                    model_path: str,
                                    config: Dict[str, Any], 
                                    model_type: str,
                                    device: str) -> ModelLoadingResult:
        """
        Load model using the new implementation with updated parameters.
        
        This method handles the actual model instantiation with the new
        parameter handling logic.
        """
        # Import here to avoid circular imports
        from ..uvr_lib_v5.roformer.bs_roformer import BSRoformer
        from ..uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
        import torch
        
        try:
            # Create model instance based on type
            if model_type == "bs_roformer":
                model = self._create_bs_roformer(config, device)
            elif model_type == "mel_band_roformer":
                model = self._create_mel_band_roformer(config, device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load state dict if model file exists
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                
                # Handle different state dict formats
                if 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                elif 'model' in state_dict:
                    model.load_state_dict(state_dict['model'])
                else:
                    model.load_state_dict(state_dict)
                
                logger.debug(f"Loaded state dict from {model_path}")
            
            model.to(device)
            model.eval()
            
            return ModelLoadingResult(
                model=model,
                model_type=model_type,
                config_used=config,
                implementation_version="new",
                loading_method="direct",
                device=device,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {e}")
            raise
    
    def _create_bs_roformer(self, config: Dict[str, Any], device: str):
        """Create BSRoformer model instance with new parameters."""
        from ..uvr_lib_v5.roformer.bs_roformer import BSRoformer
        
        # Extract parameters for BSRoformer constructor
        model_args = {
            'dim': config['dim'],
            'depth': config['depth'],
            'stereo': config.get('stereo', False),
            'num_stems': config.get('num_stems', 2),
            'time_transformer_depth': config.get('time_transformer_depth', 2),
            'freq_transformer_depth': config.get('freq_transformer_depth', 2),
            'freqs_per_bands': config['freqs_per_bands'],
            'dim_head': config.get('dim_head', 64),
            'heads': config.get('heads', 8),
            'attn_dropout': config.get('attn_dropout', 0.0),
            'ff_dropout': config.get('ff_dropout', 0.0),
            'flash_attn': config.get('flash_attn', True),
            # New parameters
            'mlp_expansion_factor': config.get('mlp_expansion_factor', 4),
            'sage_attention': config.get('sage_attention', False),
            'zero_dc': config.get('zero_dc', True),
            'use_torch_checkpoint': config.get('use_torch_checkpoint', False),
            'skip_connection': config.get('skip_connection', False),
        }
        
        # Add STFT parameters if present
        if 'stft_n_fft' in config:
            model_args['stft_n_fft'] = config['stft_n_fft']
        if 'stft_hop_length' in config:
            model_args['stft_hop_length'] = config['stft_hop_length']
        if 'stft_win_length' in config:
            model_args['stft_win_length'] = config['stft_win_length']
        
        logger.debug(f"Creating BSRoformer with args: {list(model_args.keys())}")
        return BSRoformer(**model_args)
    
    def _create_mel_band_roformer(self, config: Dict[str, Any], device: str):
        """Create MelBandRoformer model instance with new parameters."""
        from ..uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
        
        # Extract parameters for MelBandRoformer constructor
        model_args = {
            'dim': config['dim'],
            'depth': config['depth'],
            'stereo': config.get('stereo', False),
            'num_stems': config.get('num_stems', 2),
            'time_transformer_depth': config.get('time_transformer_depth', 2),
            'freq_transformer_depth': config.get('freq_transformer_depth', 2),
            'num_bands': config['num_bands'],
            'dim_head': config.get('dim_head', 64),
            'heads': config.get('heads', 8),
            'attn_dropout': config.get('attn_dropout', 0.0),
            'ff_dropout': config.get('ff_dropout', 0.0),
            'flash_attn': config.get('flash_attn', True),
            # New parameters
            'mlp_expansion_factor': config.get('mlp_expansion_factor', 4),
            'sage_attention': config.get('sage_attention', False),
            'zero_dc': config.get('zero_dc', True),
            'use_torch_checkpoint': config.get('use_torch_checkpoint', False),
            'skip_connection': config.get('skip_connection', False),
        }
        
        # Add sample rate and mel parameters if present
        if 'sample_rate' in config:
            model_args['sample_rate'] = config['sample_rate']
        if 'fmin' in config:
            model_args['fmin'] = config['fmin']
        if 'fmax' in config:
            model_args['fmax'] = config['fmax']
        
        logger.debug(f"Creating MelBandRoformer with args: {list(model_args.keys())}")
        return MelBandRoformer(**model_args)
    
    def _try_fallback_loading(self, 
                            model_path: str,
                            config: Dict[str, Any], 
                            device: str,
                            failure_reason: str) -> ModelLoadingResult:
        """
        Try loading with the fallback mechanism.
        
        Args:
            model_path: Path to model file
            config: Original configuration
            device: Device to load on
            failure_reason: Reason why primary loading failed
            
        Returns:
            ModelLoadingResult from fallback attempt
        """
        try:
            result = self.fallback_loader.try_legacy_implementation(
                model_path, config, device
            )
            
            if result.success:
                self._loading_stats['fallback_success'] += 1
                logger.info(f"Successfully loaded model with fallback implementation")
                logger.info(f"Primary failure reason: {failure_reason}")
            else:
                self._loading_stats['total_failures'] += 1
                logger.error(f"Both primary and fallback loading failed")
                
            return result
            
        except Exception as e:
            self._loading_stats['total_failures'] += 1
            logger.error(f"Fallback loading also failed: {e}")
            
            return ModelLoadingResult(
                model=None,
                model_type="unknown",
                config_used=config,
                implementation_version="fallback",
                loading_method="fallback_failed",
                device=device,
                success=False,
                error_message=f"Primary: {failure_reason}; Fallback: {e}"
            )
    
    def get_loading_stats(self) -> Dict[str, int]:
        """
        Get statistics about loading attempts.
        
        Returns:
            Dictionary with loading statistics
        """
        return self._loading_stats.copy()
    
    def reset_loading_stats(self) -> None:
        """Reset loading statistics."""
        self._loading_stats = {
            'new_implementation_success': 0,
            'fallback_success': 0,
            'total_failures': 0
        }
    
    def detect_model_type(self, model_path: str) -> str:
        """
        Detect the type of Roformer model from the file path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Detected model type string ("bs_roformer" or "mel_band_roformer")
            
        Raises:
            ValueError: If model type cannot be determined
        """
        model_path_lower = model_path.lower()
        
        # Check for BSRoformer indicators
        if any(indicator in model_path_lower for indicator in ['bs_roformer', 'bs-roformer', 'bsroformer']):
            return "bs_roformer"
        
        # Check for MelBandRoformer indicators
        if any(indicator in model_path_lower for indicator in ['mel_band_roformer', 'mel-band-roformer', 'melband']):
            return "mel_band_roformer"
        
        # Check for general roformer (default to BSRoformer)
        if 'roformer' in model_path_lower:
            logger.warning(f"Generic 'roformer' detected in {model_path}, defaulting to bs_roformer")
            return "bs_roformer"
        
        raise ValueError(f"Cannot determine Roformer model type from path: {model_path}")
    
    def get_default_configuration(self, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of Roformer model ("bs_roformer" or "mel_band_roformer")
            
        Returns:
            Default configuration dictionary for the type
        """
        if model_type == "bs_roformer":
            return {
                'dim': 512,
                'depth': 12,
                'stereo': False,
                'num_stems': 2,
                'time_transformer_depth': 2,
                'freq_transformer_depth': 2,
                'freqs_per_bands': (2, 4, 8, 16, 32, 64),
                'dim_head': 64,
                'heads': 8,
                'attn_dropout': 0.0,
                'ff_dropout': 0.0,
                'flash_attn': True,
                'mlp_expansion_factor': 4,
                'sage_attention': False,
                'zero_dc': True,
                'use_torch_checkpoint': False,
                'skip_connection': False,
                'mask_estimator_depth': 2,
                'stft_n_fft': 2048,
                'stft_hop_length': 512,
                'stft_win_length': 2048,
            }
        elif model_type == "mel_band_roformer":
            return {
                'dim': 512,
                'depth': 12,
                'stereo': False,
                'num_stems': 2,
                'time_transformer_depth': 2,
                'freq_transformer_depth': 2,
                'num_bands': 64,
                'dim_head': 64,
                'heads': 8,
                'attn_dropout': 0.0,
                'ff_dropout': 0.0,
                'flash_attn': True,
                'mlp_expansion_factor': 4,
                'sage_attention': False,
                'zero_dc': True,
                'use_torch_checkpoint': False,
                'skip_connection': False,
                'sample_rate': 44100,
                'fmin': 0,
                'fmax': None,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
