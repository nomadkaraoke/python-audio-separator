# Data Model: Roformer Implementation Update

## Core Entities

### RoformerModel
Represents a Roformer audio separation model with configuration parameters.

**Fields:**
- `model_path: str` - Path to the model file (.ckpt, .pth)
- `config: ModelConfiguration` - Model configuration parameters
- `model_type: RoformerType` - Type of Roformer (BSRoformer, MelBandRoformer)
- `implementation_version: str` - Which implementation version ("old", "new")
- `loaded_successfully: bool` - Whether model loaded without errors

**Validation Rules:**
- `model_path` must exist and be readable
- `config` must contain required parameters for the model type
- `model_type` must be one of supported variants

**State Transitions:**
- `Unloaded → Loading → Loaded` (success path)
- `Unloaded → Loading → Failed` (error path)
- `Loaded → Unloaded` (cleanup)

### ModelConfiguration
Dictionary-like object containing model architecture parameters.

**Fields:**
- `dim: int` - Model dimension
- `depth: int` - Number of transformer layers
- `stereo: bool` - Whether model handles stereo audio
- `num_stems: int` - Number of output stems
- `time_transformer_depth: int` - Depth of time transformer
- `freq_transformer_depth: int` - Depth of frequency transformer
- `freqs_per_bands: Tuple[int, ...]` - Frequency bands configuration
- `dim_head: int` - Attention head dimension
- `heads: int` - Number of attention heads
- `attn_dropout: float` - Attention dropout rate
- `ff_dropout: float` - Feed-forward dropout rate
- `flash_attn: bool` - Whether to use flash attention
- `norm: str` - Normalization type (new field handling)

**New Parameters (for updated models):**
- `mlp_expansion_factor: int = 4` - MLP expansion ratio
- `sage_attention: bool = False` - Enable Sage attention
- `zero_dc: bool = True` - Zero DC component handling
- `use_torch_checkpoint: bool = False` - Enable gradient checkpointing
- `skip_connection: bool = False` - Enable skip connections

**Validation Rules:**
- All numeric fields must be positive
- Dropout rates must be between 0.0 and 1.0
- `freqs_per_bands` must sum to expected frequency count
- `norm` must be valid normalization type or None

### BSRoformerConfig
Specialized configuration for Band-Split Roformer models.

**Fields:**
- Inherits all from `ModelConfiguration`
- `freqs_per_bands: Tuple[int, ...]` - Required, defines frequency band splits
- `mask_estimator_depth: int = 2` - Depth of mask estimation network

**Validation Rules:**
- `freqs_per_bands` must be provided and non-empty
- Sum of `freqs_per_bands` must match STFT frequency bins

### MelBandRoformerConfig  
Specialized configuration for Mel-Band Roformer models.

**Fields:**
- Inherits all from `ModelConfiguration`
- `num_bands: int` - Number of mel-scale bands
- `sample_rate: int = 44100` - Audio sample rate for mel calculation

**Validation Rules:**
- `num_bands` must be positive integer
- `sample_rate` must be valid audio sample rate

### ModelLoadingResult
Result object returned from model loading attempts.

**Fields:**
- `success: bool` - Whether loading succeeded
- `model: Optional[RoformerModel]` - Loaded model if successful
- `error_message: Optional[str]` - Error description if failed
- `implementation_used: str` - Which implementation was used ("old", "new", "fallback")
- `warnings: List[str]` - Non-fatal warnings during loading

**State Transitions:**
- `Attempting → Success` (with model)
- `Attempting → Failure` (with error message)
- `Success → Cleanup` (model disposal)

### ParameterValidationError
Exception raised when model parameters are invalid.

**Fields:**
- `parameter_name: str` - Name of invalid parameter
- `expected_type: str` - Expected parameter type
- `actual_value: Any` - Actual value provided
- `suggested_fix: str` - Suggestion for fixing the issue

## Entity Relationships

```
RoformerModel
├── has_one ModelConfiguration
│   ├── extends_to BSRoformerConfig
│   └── extends_to MelBandRoformerConfig
├── produces ModelLoadingResult
└── may_raise ParameterValidationError

ModelConfiguration
├── validates_against ValidationRules
└── contains ParameterSet
    ├── required_parameters
    ├── optional_parameters
    └── new_parameters
```

## Data Flow

### Model Loading Flow
1. `ModelLoader.load(model_path)` → `ModelLoadingResult`
2. Parse configuration from model file
3. Validate configuration parameters
4. Attempt loading with new implementation
5. On failure, attempt loading with old implementation
6. Return result with success/failure status

### Parameter Validation Flow
1. Extract parameters from model configuration
2. Check required parameters exist
3. Validate parameter types and ranges
4. Check parameter compatibility
5. Raise `ParameterValidationError` with specific details if invalid

### Configuration Normalization Flow
1. Load raw configuration from model
2. Map old parameter names to new parameter names
3. Set default values for missing optional parameters
4. Validate final configuration
5. Return normalized `ModelConfiguration`

## Storage Considerations

- Model files are read-only external artifacts
- Configuration is derived from model metadata
- No persistent state storage required
- Memory usage scales with model size
- Cleanup required after model disposal

## Performance Considerations

- Model loading is I/O intensive operation
- Configuration validation should be fast
- Parameter defaults should be computed once
- Error messages should be pre-formatted for common cases
- Fallback mechanism adds latency but ensures compatibility
