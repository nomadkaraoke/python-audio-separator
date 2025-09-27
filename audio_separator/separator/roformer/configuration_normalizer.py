"""
Configuration normalizer for Roformer models.
Normalizes and standardizes configuration dictionaries from various sources.
"""

from typing import Dict, Any, Optional, Union, List
import copy
import logging

from .parameter_validator import ParameterValidator
from .bs_roformer_validator import BSRoformerValidator
from .mel_band_roformer_validator import MelBandRoformerValidator
from .parameter_validation_error import ParameterValidationError

logger = logging.getLogger(__name__)


class ConfigurationNormalizer:
    """
    Normalizes configuration dictionaries for Roformer models.
    
    Handles different configuration formats, applies defaults, and ensures
    consistency across different model types and versions.
    """
    
    def __init__(self):
        """Initialize the configuration normalizer with validators."""
        self.base_validator = ParameterValidator()
        self.bs_validator = BSRoformerValidator()
        self.mel_validator = MelBandRoformerValidator()
    
    def normalize_config(self, 
                        config: Dict[str, Any], 
                        model_type: str, 
                        apply_defaults: bool = True,
                        validate: bool = True) -> Dict[str, Any]:
        """
        Normalize a configuration dictionary.
        
        Args:
            config: Raw configuration dictionary
            model_type: Type of model ("bs_roformer" or "mel_band_roformer")
            apply_defaults: Whether to apply default values for missing parameters
            validate: Whether to validate the configuration
            
        Returns:
            Normalized configuration dictionary
            
        Raises:
            ParameterValidationError: If validation fails and validate=True
        """
        # Deep copy to avoid modifying original
        normalized = copy.deepcopy(config)
        
        # Step 1: Normalize structure (flatten nested configs if needed)
        normalized = self._normalize_structure(normalized, model_type)
        
        # Step 2: Normalize parameter names and values
        normalized = self._normalize_parameter_names(normalized)
        normalized = self._normalize_parameter_values(normalized, model_type)
        
        # Step 3: Apply defaults if requested
        if apply_defaults:
            normalized = self._apply_defaults(normalized, model_type)
        
        # Step 4: Validate if requested
        if validate:
            self._validate_config(normalized, model_type)
        
        logger.debug(f"Normalized {model_type} configuration: {len(normalized)} parameters")
        return normalized
    
    def _normalize_structure(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Normalize the structure of the configuration dictionary.
        
        Some configurations may be nested or have different structures.
        This flattens and standardizes the structure.
        """
        normalized = {}
        
        # Handle nested configurations (e.g., from YAML files)
        for key, value in config.items():
            if isinstance(value, dict) and key in ['model', 'architecture', 'params']:
                # Flatten nested model parameters
                normalized.update(value)
            elif key in ['training', 'inference'] and isinstance(value, dict):
                # Some configs have training/inference sections
                # Extract relevant parameters with prefixes
                for nested_key, nested_value in value.items():
                    if nested_key in ['dim_t', 'hop_length', 'n_fft', 'sample_rate']:
                        normalized[nested_key] = nested_value
            else:
                normalized[key] = value
        
        return normalized
    
    def _normalize_parameter_names(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameter names to standard format.
        
        Handles different naming conventions and aliases.
        """
        normalized = {}
        
        # Parameter name mappings (old_name -> new_name)
        name_mappings = {
            # Common aliases
            'n_fft': 'stft_n_fft',
            'hop_length': 'stft_hop_length', 
            'win_length': 'stft_win_length',
            'window_fn': 'stft_window_fn',
            'normalized': 'stft_normalized',
            
            # Transformer aliases
            'n_heads': 'heads',
            'num_heads': 'heads',
            'head_dim': 'dim_head',
            'dropout': 'attn_dropout',
            'attention_dropout': 'attn_dropout',
            'feedforward_dropout': 'ff_dropout',
            
            # Model-specific aliases
            'expansion_factor': 'mlp_expansion_factor',
            'mlp_ratio': 'mlp_expansion_factor',
            'use_checkpoint': 'use_torch_checkpoint',
            'checkpoint': 'use_torch_checkpoint',
            
            # Frequency band aliases
            'freq_bands': 'freqs_per_bands',
            'frequency_bands': 'freqs_per_bands',
            'mel_bands': 'num_bands',
            'n_mels': 'num_bands',
        }
        
        for key, value in config.items():
            # Apply name mapping if exists
            normalized_key = name_mappings.get(key, key)
            normalized[normalized_key] = value
        
        return normalized
    
    def _normalize_parameter_values(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Normalize parameter values to expected types and formats.
        """
        normalized = {}
        
        for key, value in config.items():
            normalized_value = self._normalize_single_value(key, value, model_type)
            normalized[key] = normalized_value
        
        return normalized
    
    def _normalize_single_value(self, key: str, value: Any, model_type: str) -> Any:
        """Normalize a single parameter value."""
        
        # Boolean normalization
        if key in ['stereo', 'flash_attn', 'sage_attention', 'zero_dc', 
                  'use_torch_checkpoint', 'skip_connection', 'stft_normalized']:
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        
        # Integer normalization
        elif key in ['dim', 'depth', 'num_stems', 'time_transformer_depth', 
                    'freq_transformer_depth', 'dim_head', 'heads', 
                    'mlp_expansion_factor', 'num_bands', 'sample_rate',
                    'stft_n_fft', 'stft_hop_length', 'stft_win_length',
                    'mask_estimator_depth']:
            if isinstance(value, str):
                try:
                    return int(float(value))  # Handle "2.0" -> 2
                except (ValueError, TypeError):
                    return value  # Let validator catch the error
            return int(value) if isinstance(value, (int, float)) else value
        
        # Float normalization  
        elif key in ['attn_dropout', 'ff_dropout', 'multi_stft_resolution_loss_weight',
                    'fmin', 'fmax']:
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value  # Let validator catch the error
            return float(value) if isinstance(value, (int, float)) else value
        
        # Tuple/list normalization
        elif key.startswith('freqs_per_bands') or key in ['freqs_per_bands']:
            if isinstance(value, str):
                # Handle string representations like "(2, 4, 8, 16)"
                try:
                    # Remove parentheses and split
                    clean_str = value.strip('()[]').replace(' ', '')
                    if clean_str:
                        return tuple(int(x) for x in clean_str.split(','))
                except (ValueError, TypeError):
                    return value  # Let validator catch the error
            elif isinstance(value, list):
                return tuple(value)  # Convert lists to tuples for consistency
            return value
        
        # String normalization
        elif key in ['norm', 'act', 'mel_scale']:
            if value is not None:
                return str(value).lower()
            return value
        
        # No normalization needed
        else:
            return value
    
    def _apply_defaults(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Apply default values for missing parameters."""
        
        if model_type == "bs_roformer":
            validator = self.bs_validator
        elif model_type == "mel_band_roformer":
            validator = self.mel_validator
        else:
            validator = self.base_validator
        
        return validator.apply_parameter_defaults(config, model_type)
    
    def _validate_config(self, config: Dict[str, Any], model_type: str) -> None:
        """Validate the configuration and raise errors if invalid."""
        
        if model_type == "bs_roformer":
            validator = self.bs_validator
        elif model_type == "mel_band_roformer":
            validator = self.mel_validator
        else:
            validator = self.base_validator
        
        validator.validate_and_raise(config, model_type)
    
    def detect_model_type(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Attempt to detect the model type from configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Detected model type or None if cannot be determined
        """
        # Check for BSRoformer-specific parameters
        if 'freqs_per_bands' in config:
            return "bs_roformer"
        
        # Check for MelBandRoformer-specific parameters
        if 'num_bands' in config or 'n_mels' in config or 'mel_bands' in config:
            return "mel_band_roformer"
        
        # Check for model type hints in the config
        model_type = config.get('model_type', config.get('type', config.get('architecture')))
        if isinstance(model_type, str):
            model_type_lower = model_type.lower()
            if 'bs' in model_type_lower and 'roformer' in model_type_lower:
                return "bs_roformer"
            elif 'mel' in model_type_lower and 'roformer' in model_type_lower:
                return "mel_band_roformer"
            elif 'roformer' in model_type_lower:
                # Default to BSRoformer if just "roformer"
                return "bs_roformer"
        
        return None
    
    def normalize_from_file_path(self, 
                                config: Dict[str, Any], 
                                file_path: str,
                                apply_defaults: bool = True,
                                validate: bool = True) -> Dict[str, Any]:
        """
        Normalize configuration with model type detection from file path.
        
        Args:
            config: Configuration dictionary
            file_path: Path to the model file (used for type detection)
            apply_defaults: Whether to apply defaults
            validate: Whether to validate
            
        Returns:
            Normalized configuration
        """
        # Try to detect model type from file path
        file_path_lower = file_path.lower()
        if 'bs' in file_path_lower and 'roformer' in file_path_lower:
            model_type = "bs_roformer"
        elif 'mel' in file_path_lower and 'roformer' in file_path_lower:
            model_type = "mel_band_roformer"
        else:
            # Try to detect from config
            model_type = self.detect_model_type(config)
            if model_type is None:
                # Default to BSRoformer
                model_type = "bs_roformer"
                logger.warning(f"Could not detect model type from config or path {file_path}, defaulting to bs_roformer")
        
        return self.normalize_config(config, model_type, apply_defaults, validate)
