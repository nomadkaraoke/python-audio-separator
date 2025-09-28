# Roformer Implementation - Updated Parameter Support

This directory contains the updated Roformer implementation with support for new parameters and backward compatibility for legacy models (handled by the single, modern loader).

## Overview

The updated Roformer implementation provides:

- **New Parameter Support**: `mlp_expansion_factor`, `sage_attention`, `zero_dc`, `use_torch_checkpoint`, `skip_connection`
- **Single Loader Path**: Unified loader supports both newer and older checkpoints
- **Robust Validation**: Comprehensive parameter validation with detailed error messages
- **Configuration Normalization**: Handles different config formats and applies sensible defaults

## Architecture

```
roformer/
├── model_configuration.py      # Data model for standardized config
├── parameter_validator.py      # Base parameter validation
├── bs_roformer_validator.py    # BSRoformer-specific validation
├── mel_band_roformer_validator.py  # MelBandRoformer-specific validation
├── configuration_normalizer.py # Config normalization and defaults
├── parameter_validation_error.py  # Custom exception handling
├── roformer_loader.py          # Main loader (new implementation only)
├── model_loading_result.py     # Result dataclass
└── README.md                   # This documentation
```

## New Parameters

### Core New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mlp_expansion_factor` | int | 4 | MLP expansion factor in transformer layers |
| `sage_attention` | bool | False | Use Sage attention mechanism instead of standard attention |
| `zero_dc` | bool | True | Apply zero DC component filtering |
| `use_torch_checkpoint` | bool | False | Enable gradient checkpointing for memory efficiency |
| `skip_connection` | bool | False | Add skip connections in the architecture |

### Compatibility Notes

- **`sage_attention`**: Cannot be used simultaneously with `flash_attn=True` (warning will be issued)
- **`mlp_expansion_factor`**: Higher values increase model capacity but also memory usage
- **`use_torch_checkpoint`**: Trades computation for memory - useful for large models
- **`zero_dc`**: Recommended for most audio separation tasks

## Usage Examples

### Basic Usage with New Parameters

```python
from audio_separator.separator.roformer.roformer_loader import RoformerLoader

loader = RoformerLoader()

config = {
    'dim': 512,
    'depth': 12,
    'freqs_per_bands': (2, 4, 8, 16, 32, 64),
    'mlp_expansion_factor': 8,
    'sage_attention': True,
    'zero_dc': True,
    'use_torch_checkpoint': True,
    'skip_connection': False
}

result = loader.load_model('/path/to/model.ckpt', config, device='cuda')
if result.success:
    print("Model loaded")
else:
    print(f"Failed: {result.error_message}")
```

### Configuration Validation

```python
from audio_separator.separator.roformer.parameter_validator import ParameterValidator

validator = ParameterValidator()
issues = validator.validate_all(config, "bs_roformer")
if issues:
    for issue in issues:
        print(f"{issue.severity.value}: {issue.message}")
else:
    print("Configuration is valid!")
```

### Configuration Normalization

```python
from audio_separator.separator.roformer.configuration_normalizer import ConfigurationNormalizer

normalizer = ConfigurationNormalizer()
raw_config = {
    'model': {'dim': '512', 'depth': 12.0, 'n_heads': 8},
    'training': {'sample_rate': 44100},
    'stereo': 'true',
    'freq_bands': '[2, 4, 8, 16]'
}
normalized = normalizer.normalize_config(raw_config, model_type="bs_roformer", apply_defaults=True, validate=True)
print(f"Normalized config has {len(normalized)} parameters")
```

## Model Type Detection

Detected via config contents (e.g., `freqs_per_bands` vs `num_bands`) and filename hints. Defaults to BSRoformer when ambiguous.

## Integration with Audio Separator

- Routing remains through the MDXC architecture path; Roformer models are detected and handled by the MDXC separator using the unified `RoformerLoader`.
- Loader statistics are surfaced via `CommonSeparator.get_roformer_loading_stats()` and logged by the top-level `Separator`.

## Testing

```bash
# Unit tests (examples)
python -m pytest tests/unit/test_parameter_validator.py -v
python -m pytest tests/unit/test_configuration_normalizer.py -v

# Integration
python -m pytest tests/integration/test_roformer_*.py -v
```

## Contributing

- Follow TDD
- Maintain compatibility for existing checkpoints through the single loader path
- Update documentation when adding parameters or behavior

## License

Follows the main Audio Separator project license.
