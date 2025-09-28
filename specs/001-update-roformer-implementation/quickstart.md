# Quickstart Guide: Updated Roformer Implementation

This guide demonstrates how to use the updated Roformer implementation with both old and new models.

## Prerequisites

- Python 3.11+
- PyTorch installed
- audio-separator package installed
- Access to Roformer model files (.ckpt or .pth)

## Basic Usage

### Loading a Roformer Model

```python
from audio_separator import Separator

# Initialize separator with Roformer model
separator = Separator(
    model_file_dir='path/to/models',
    output_dir='path/to/output'
)

# Load a Roformer model (automatically detects old vs new format)
separator.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
```

### Separating Audio

```python
# Separate audio file
output_files = separator.separate('input_audio.flac')

print(f"Separation complete. Output files: {output_files}")
```

## Model Types

### BSRoformer Models

```python
# BSRoformer models work with frequency band splitting
separator.load_model('bs_roformer_model.ckpt')
outputs = separator.separate('audio.flac')
# Outputs: ['audio_(Vocals).flac', 'audio_(Instrumental).flav']
```

### MelBandRoformer Models  

```python
# MelBandRoformer models work with mel-scale bands
separator.load_model('mel_band_roformer_model.ckpt')
outputs = separator.separate('audio.flac')
# Outputs depend on model configuration
```

## Advanced Configuration

### Manual Configuration Override

```python
# Override model configuration if needed
config_override = {
    'mlp_expansion_factor': 4,
    'sage_attention': True,
    'zero_dc': True,
    'use_torch_checkpoint': False
}

separator.load_model('model.ckpt', config=config_override)
```

### Error Handling

```python
try:
    separator.load_model('problematic_model.ckpt')
    outputs = separator.separate('audio.flac')
except ParameterValidationError as e:
    print(f"Configuration error: {e}")
    print(f"Suggestion: {e.suggested_fix}")
except Exception as e:
    print(f"Loading failed: {e}")
```

## Testing Scenarios

### Scenario 1: Existing Older Model

Test that existing models continue to work without changes.

```python
# This should work exactly as before
separator = Separator()
separator.load_model('old_roformer_model.ckpt')
outputs = separator.separate('test_audio.flac')

# Verify outputs match previous results
assert len(outputs) == 2  # Expecting vocal and instrumental
assert outputs[0].endswith('_(Vocals).flav')
assert outputs[1].endswith('_(Instrumental).flav')
```

### Scenario 2: Newer Model with New Parameters

Test that newer models with additional parameters work correctly.

```python
# This should work with the updated implementation
separator = Separator()
separator.load_model('new_roformer_with_sage_attention.ckpt')
outputs = separator.separate('test_audio.flac')

# Verify outputs are generated successfully
assert len(outputs) >= 1
for output in outputs:
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0
```

### Scenario 3: Model Type Switching

Test switching between different Roformer variants in the same session.

```python
separator = Separator()

# Load BSRoformer model
separator.load_model('bs_roformer.ckpt')
bs_outputs = separator.separate('test1.flac')

# Switch to MelBandRoformer model  
separator.load_model('mel_band_roformer.ckpt')
mel_outputs = separator.separate('test2.flac')

# Both should work without conflicts
assert len(bs_outputs) > 0
assert len(mel_outputs) > 0
```

### Scenario 4: Configuration Validation

Test that invalid configurations are caught with helpful error messages.

```python
# Test missing required parameter
try:
    separator.load_model('model_with_missing_config.ckpt')
    assert False, "Should have raised validation error"
except ParameterValidationError as e:
    assert "missing" in e.suggested_fix.lower()
    assert e.parameter_name is not None
```

### Scenario 5: Fallback Mechanism

Test that fallback from new to old implementation works.

```python
# This would internally try new implementation first, then fallback to old
separator = Separator(log_level='DEBUG')  # Enable debug logging
separator.load_model('edge_case_model.ckpt')

# Check logs to verify fallback occurred if needed
# (Implementation should log which version was used)
```

## Performance Validation

### Audio Quality Validation

```python
import subprocess
import os

# Run integration test to validate audio quality
result = subprocess.run([
    'python', '-m', 'pytest', 
    'tests/integration/test_roformer_quality.py',
    '-v'
], capture_output=True, text=True)

assert result.returncode == 0, f"Quality tests failed: {result.stderr}"
```

### Regression Testing

```python
# Ensure existing models produce identical results
reference_outputs = load_reference_outputs('test_audio.flac')
current_outputs = separator.separate('test_audio.flav')

for ref, curr in zip(reference_outputs, current_outputs):
    similarity = calculate_audio_similarity(ref, curr)
    assert similarity >= 0.90, f"Audio similarity {similarity} below threshold"
```

## Troubleshooting

### Common Issues

1. **AttributeError: 'norm'**
   - Cause: Model configuration has different normalization structure
   - Solution: The updated implementation handles this automatically

2. **TypeError: unexpected keyword argument 'mlp_expansion_factor'**
   - Cause: Trying to use new model with old implementation
   - Solution: The fallback mechanism handles this automatically

3. **Model loading fails completely**
   - Check model file exists and is readable
   - Verify model is a supported Roformer variant
   - Check logs for specific error details

### Debug Information

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

separator = Separator()
separator.load_model('model.ckpt')
# Check logs for implementation version used and any warnings
```

## Manual Testing Checklist

After implementation changes, manually verify:

- [ ] Existing BSRoformer models load and separate correctly
- [ ] Existing MelBandRoformer models load and separate correctly  
- [ ] New models with additional parameters work
- [ ] Audio quality matches reference outputs (SSIM ≥ 0.90/0.80)
- [ ] Error messages are clear and helpful
- [ ] Performance is not significantly degraded
- [ ] Memory usage is reasonable
- [ ] Multiple models can be loaded in sequence

## Next Steps

After validating the quickstart scenarios:

1. Run full integration test suite
2. Perform manual testing with real model files
3. Validate performance benchmarks
4. Update documentation if needed
5. Prepare for production deployment
