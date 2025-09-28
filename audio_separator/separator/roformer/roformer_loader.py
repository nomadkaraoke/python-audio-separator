"""Roformer model loader with simplified new-implementation only path."""
from typing import Dict, Any
import logging
import os

from .model_loading_result import ModelLoadingResult, ImplementationVersion
from .configuration_normalizer import ConfigurationNormalizer
from .parameter_validation_error import ParameterValidationError

logger = logging.getLogger(__name__)


class RoformerLoader:
    """Main Roformer model loader (new implementation only)."""

    def __init__(self):
        self.config_normalizer = ConfigurationNormalizer()
        self._loading_stats = {
            'new_implementation_success': 0,
            'total_failures': 0
        }

    def load_model(self,
                   model_path: str,
                   config: Dict[str, Any],
                   device: str = 'cpu') -> ModelLoadingResult:
        logger.info(f"Loading Roformer model from {model_path}")
        try:
            normalized_config = self.config_normalizer.normalize_from_file_path(
                config, model_path, apply_defaults=True, validate=True
            )
            model_type = self.config_normalizer.detect_model_type(normalized_config)
            logger.debug(f"Detected model type: {model_type}")
        except ParameterValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            return ModelLoadingResult.failure_result(
                error_message=f"Config validation: {e}",
                implementation=ImplementationVersion.NEW,
            )

        try:
            result = self._load_with_new_implementation(
                model_path, normalized_config, model_type, device
            )
            self._loading_stats['new_implementation_success'] += 1
            logger.info(f"Successfully loaded {model_type} model with new implementation")
            return result
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"New implementation failed: {e}")
            # Attempt legacy fallback using the original (pre-normalized) configuration
            try:
                fallback_result = self._load_with_legacy_implementation(
                    model_path=model_path,
                    original_config=config,
                    device=device,
                    original_error=str(e)
                )
                logger.warning("Fell back to legacy Roformer implementation successfully")
                return fallback_result
            except (RuntimeError, ValueError, TypeError) as fallback_error:
                logger.error(f"Legacy implementation also failed: {fallback_error}")
                self._loading_stats['total_failures'] += 1
                return ModelLoadingResult.failure_result(
                    error_message=f"New implementation failed: {e}; Legacy fallback failed: {fallback_error}",
                    implementation=ImplementationVersion.NEW,
                )

    def validate_configuration(self, config: Dict[str, Any], model_type: str) -> bool:
        try:
            _ = self.config_normalizer.normalize_config(
                config, model_type, apply_defaults=False, validate=True
            )
            logger.debug(f"Configuration validation passed for {model_type}")
            return True
        except ParameterValidationError as e:
            logger.warning(f"Configuration validation failed for {model_type}: {e}")
            return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False

    def _load_with_new_implementation(self,
                                      model_path: str,
                                      config: Dict[str, Any],
                                      model_type: str,
                                      device: str) -> ModelLoadingResult:
        import torch

        try:
            if model_type == "bs_roformer":
                model = self._create_bs_roformer(config)
            elif model_type == "mel_band_roformer":
                model = self._create_mel_band_roformer(config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                elif isinstance(state_dict, dict) and 'model' in state_dict:
                    model.load_state_dict(state_dict['model'])
                else:
                    model.load_state_dict(state_dict)
                logger.debug(f"Loaded state dict from {model_path}")

            model.to(device)
            model.eval()

            result = ModelLoadingResult.success_result(
                model=model,
                implementation=ImplementationVersion.NEW,
                config=config,
            )
            result.add_model_info('model_type', model_type)
            result.add_model_info('loading_method', 'direct')
            result.add_model_info('device', device)
            return result
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to create {model_type} model: {e}")
            raise

    def _create_bs_roformer(self, config: Dict[str, Any]):
        from ..uvr_lib_v5.roformer.bs_roformer import BSRoformer
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
            'mlp_expansion_factor': config.get('mlp_expansion_factor', 4),
            'sage_attention': config.get('sage_attention', False),
            'zero_dc': config.get('zero_dc', True),
            'use_torch_checkpoint': config.get('use_torch_checkpoint', False),
            'skip_connection': config.get('skip_connection', False),
        }
        if 'stft_n_fft' in config:
            model_args['stft_n_fft'] = config['stft_n_fft']
        if 'stft_hop_length' in config:
            model_args['stft_hop_length'] = config['stft_hop_length']
        if 'stft_win_length' in config:
            model_args['stft_win_length'] = config['stft_win_length']
        logger.debug(f"Creating BSRoformer with args: {list(model_args.keys())}")
        return BSRoformer(**model_args)

    def _create_mel_band_roformer(self, config: Dict[str, Any]):
        from ..uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
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
            'mlp_expansion_factor': config.get('mlp_expansion_factor', 4),
            'sage_attention': config.get('sage_attention', False),
            'zero_dc': config.get('zero_dc', True),
            'use_torch_checkpoint': config.get('use_torch_checkpoint', False),
            'skip_connection': config.get('skip_connection', False),
        }
        if 'sample_rate' in config:
            model_args['sample_rate'] = config['sample_rate']
        # Optional parameters commonly present in legacy configs
        for optional_key in [
            'mask_estimator_depth',
            'stft_n_fft',
            'stft_hop_length',
            'stft_win_length',
            'stft_normalized',
            'stft_window_fn',
            'multi_stft_resolution_loss_weight',
            'multi_stft_resolutions_window_sizes',
            'multi_stft_hop_size',
            'multi_stft_normalized',
            'multi_stft_window_fn',
            'match_input_audio_length',
        ]:
            if optional_key in config:
                model_args[optional_key] = config[optional_key]
        # Note: fmin and fmax are defined in config classes but not accepted by current constructor
        logger.debug(f"Creating MelBandRoformer with args: {list(model_args.keys())}")
        return MelBandRoformer(**model_args)

    def _load_with_legacy_implementation(self,
                                          model_path: str,
                                          original_config: Dict[str, Any],
                                          device: str,
                                          original_error: str) -> ModelLoadingResult:
        """
        Attempt to load the model using the legacy direct-constructor path
        for maximum backward compatibility with existing checkpoints.
        """
        import torch

        # Use nested 'model' section if present; otherwise assume flat
        model_cfg = original_config.get('model', original_config)

        # Determine model type from config
        if 'num_bands' in model_cfg:
            from ..uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
            model = MelBandRoformer(**model_cfg)
        elif 'freqs_per_bands' in model_cfg:
            from ..uvr_lib_v5.roformer.bs_roformer import BSRoformer
            model = BSRoformer(**model_cfg)
        else:
            raise ValueError("Unknown Roformer model type in legacy configuration")

        # Load checkpoint as raw state dict (legacy behavior)
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            # For older torch versions without weights_only
            checkpoint = torch.load(model_path, map_location='cpu')

        model.load_state_dict(checkpoint)
        model.to(device).eval()

        return ModelLoadingResult.fallback_success_result(
            model=model,
            original_error=original_error,
            config=original_config,
        )

    def get_loading_stats(self) -> Dict[str, int]:
        return self._loading_stats.copy()

    def reset_loading_stats(self) -> None:
        self._loading_stats = {
            'new_implementation_success': 0,
            'total_failures': 0
        }

    def detect_model_type(self, model_path: str) -> str:
        model_path_lower = model_path.lower()
        if any(indicator in model_path_lower for indicator in ['bs_roformer', 'bs-roformer', 'bsroformer']):
            return "bs_roformer"
        if any(indicator in model_path_lower for indicator in ['mel_band_roformer', 'mel-band-roformer', 'melband']):
            return "mel_band_roformer"
        if 'roformer' in model_path_lower:
            logger.warning(f"Generic 'roformer' detected in {model_path}, defaulting to bs_roformer")
            return "bs_roformer"
        raise ValueError(f"Cannot determine Roformer model type from path: {model_path}")

    def get_default_configuration(self, model_type: str) -> Dict[str, Any]:
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
                # Note: fmin and fmax are not implemented in MelBandRoformer constructor
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
