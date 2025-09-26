# Tasks: Update Roformer Implementation

**Input**: Design documents from `/specs/001-update-roformer-implementation/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: Python 3.11+, PyTorch, librosa, soundfile, numpy, onnxruntime
   → Structure: single project (audio_separator/ library with CLI wrapper)
2. Load design documents:
   → data-model.md: 6 entities → model tasks
   → contracts/: 2 interface files → contract test tasks
   → quickstart.md: 5 test scenarios → integration test tasks
3. Generate tasks by category:
   → Setup: dependencies, linting, project structure
   → Tests: contract tests, integration tests (TDD)
   → Core: models, parameter validation, fallback mechanism
   → Integration: existing separator integration, CLI updates
   → Polish: unit tests, performance validation, docs
4. Apply TDD ordering: Tests before implementation
5. Mark [P] for parallel execution (different files)
6. Validate: All contracts tested, all entities implemented
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup

- [x] T001 Update project dependencies in pyproject.toml to ensure PyTorch compatibility for new Roformer parameters
- [x] T002 [P] Configure linting rules in pyproject.toml for new Roformer implementation files
- [x] T003 [P] Create backup of existing uvr_lib_v5/roformer/ directory for rollback safety

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests
- [x] T004 [P] Contract test RoformerLoaderInterface.load_model in tests/contract/test_roformer_loader_interface.py
- [x] T005 [P] Contract test RoformerLoaderInterface.validate_configuration in tests/contract/test_roformer_loader_interface.py
- [x] T006 [P] Contract test ParameterValidatorInterface.validate_required_parameters in tests/contract/test_parameter_validator_interface.py
- [x] T007 [P] Contract test ParameterValidatorInterface.validate_normalization_config in tests/contract/test_parameter_validator_interface.py
- [x] T008 [P] Contract test FallbackLoaderInterface.try_new_implementation in tests/contract/test_fallback_loader_interface.py

### Integration Tests (from quickstart scenarios)
- [x] T009 [P] Integration test existing older model compatibility in tests/integration/test_roformer_backward_compatibility.py
- [x] T010 [P] Integration test newer model with new parameters in tests/integration/test_roformer_new_parameters.py
- [x] T011 [P] Integration test model type switching (BSRoformer ↔ MelBandRoformer) in tests/integration/test_roformer_model_switching.py
- [x] T012 [P] Integration test configuration validation error handling in tests/integration/test_roformer_config_validation.py
- [x] T013 [P] Integration test fallback mechanism activation in tests/integration/test_roformer_fallback_mechanism.py

### Audio Quality Validation Tests
- [x] T014 [P] Audio quality regression test for existing BSRoformer models in tests/integration/test_roformer_audio_quality.py
- [x] T015 [P] Audio quality validation test for MelBandRoformer models in tests/integration/test_roformer_audio_quality.py

### Roformer Chunking and Overlap Unit Tests
- [x] T051 [P] Assert chunk_size uses model.stft_hop_length: tests/unit/test_mdxc_roformer_chunking.py::test_chunk_size_uses_model_stft_hop_length
- [x] T052 [P] Fallback to audio.hop_length if model.stft_hop_length missing: tests/unit/test_mdxc_roformer_chunking.py::test_chunk_size_falls_back_to_audio_hop_length
- [x] T053 [P] Step clamped to chunk_size (desired_step > chunk_size or ≤ 0): tests/unit/test_mdxc_roformer_chunking.py::test_step_clamped_to_chunk_size
- [x] T054 [P] overlap_add handles shorter model output safely (safe_len): tests/unit/test_mdxc_roformer_chunking.py::test_overlap_add_short_output_safe
- [x] T055 [P] Counter increments match overlap_add safe span: tests/unit/test_mdxc_roformer_chunking.py::test_counter_updates_safe_len
- [x] T056 [P] No NaN/inf on normalization (counter clamp): tests/unit/test_mdxc_roformer_chunking.py::test_counter_clamp_no_nan
- [x] T057 [P] Short-audio last-block path works and preserves length: tests/unit/test_mdxc_roformer_chunking.py::test_short_audio_last_block
- [x] T058 [P] Parametrized invariants across dim_t and hop configs: tests/unit/test_mdxc_roformer_chunking.py::test_parametrized_shape_invariants

### Roformer Detection Tests
- [x] T059 [P] YAML path containing "roformer" sets is_roformer and routes Roformer path: tests/unit/test_separator_detection.py::test_is_roformer_set_from_yaml_path

### Roformer Integration (E2E)
- [x] T060 [P] New BSRoformer SW-Fixed end-to-end separation succeeds: tests/integration/test_roformer_e2e.py::test_bs_roformer_sw_fixed_e2e
- [x] T061 [P] Legacy Roformer end-to-end separation still succeeds: tests/integration/test_roformer_e2e.py::test_legacy_roformer_e2e

### Regression: Size mismatch family
- [x] T062 [P] Reproduce shorter outputs (N∈{1,16,32,236}) and assert no broadcast errors, output length preserved: tests/regression/test_roformer_size_mismatch.py::test_overlap_add_safe_lengths

### Logging and Performance Sanity
- [x] T063 [P] Logs include hop/step sources (stft_hop_length, dim_t, desired vs actual step): tests/unit/test_mdxc_roformer_chunking.py::test_logging_for_hop_and_step
- [x] T064 [P] Iteration count reasonable (ceil calculation within ±1): tests/unit/test_mdxc_roformer_chunking.py::test_iteration_count_reasonable

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models and Configuration
- [x] T016 [P] Implement ModelConfiguration dataclass in audio_separator/separator/roformer/model_configuration.py
- [x] T017 [P] Implement BSRoformerConfig class in audio_separator/separator/roformer/bs_roformer_config.py
- [x] T018 [P] Implement MelBandRoformerConfig class in audio_separator/separator/roformer/mel_band_roformer_config.py
- [x] T019 [P] Implement ModelLoadingResult dataclass in audio_separator/separator/roformer/model_loading_result.py
- [x] T020 [P] Implement ParameterValidationError exception in audio_separator/separator/roformer/parameter_validation_error.py

### Parameter Validation System
- [x] T021 [P] Implement ParameterValidator class in audio_separator/separator/roformer/parameter_validator.py
- [x] T022 [P] Implement BSRoformerValidator class in audio_separator/separator/roformer/bs_roformer_validator.py
- [x] T023 [P] Implement MelBandRoformerValidator class in audio_separator/separator/roformer/mel_band_roformer_validator.py
- [x] T024 [P] Implement ConfigurationNormalizer class in audio_separator/separator/roformer/configuration_normalizer.py

### Updated Roformer Models
- [x] T025 Update BSRoformer.__init__ method in audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py to add new parameters (mlp_expansion_factor, sage_attention, zero_dc, use_torch_checkpoint, skip_connection)
- [x] T026 Update MelBandRoformer.__init__ method in audio_separator/separator/uvr_lib_v5/roformer/mel_band_roformer.py to add new parameters
- [x] T027 Update transformer_kwargs in BSRoformer to include sage_attention parameter
- [x] T028 Update transformer_kwargs in MelBandRoformer to include sage_attention parameter

### Normalization Fixes
- [x] T029 Update get_norm function in audio_separator/separator/uvr_lib_v5/tfc_tdf_v3.py to handle missing norm attributes gracefully
- [x] T030 Add defensive checks in TFC_TDF_net.__init__ for normalization configuration

### Fallback Loading System
- [x] T031 [P] Implement RoformerLoader class with fallback mechanism in audio_separator/separator/roformer/roformer_loader.py
- [x] T032 [P] Implement FallbackLoader class in audio_separator/separator/roformer/fallback_loader.py
- [x] T033 Update RoformerSeparator class in audio_separator/separator/architectures/roformer_separator.py to use new loading system

## Phase 3.4: Integration

- [x] T034 Integrate new RoformerLoader into CommonSeparator base class in audio_separator/separator/common_separator.py
- [x] T035 Update CLI model loading logic to use new fallback mechanism in audio_separator/separator/separator.py
- [x] T036 Add logging for implementation version used (old/new/fallback) in audio_separator/separator/roformer/roformer_loader.py
- [x] T037 Update error handling and user messages for configuration validation failures
- [x] T038 Integrate with existing integration test framework (test_cli_integration.py compatibility)

## Phase 3.5: Polish

### Unit Tests
- [x] T039 [P] Unit tests for ModelConfiguration validation in tests/unit/test_model_configuration.py
- [x] T040 [P] Unit tests for ParameterValidator methods in tests/unit/test_parameter_validator.py
- [x] T041 [P] Unit tests for ConfigurationNormalizer methods in tests/unit/test_configuration_normalizer.py
- [x] T042 [P] Unit tests for fallback mechanism logic in tests/unit/test_fallback_loader.py

### Performance and Quality Validation
- [x] T043 Run existing integration tests to ensure no regression in audio quality (SSIM ≥ 0.90/0.80)
- [x] T044 Performance benchmark for model loading time (should not significantly increase)
- [x] T045 Memory usage validation for new parameter handling
- [x] T046 [P] Update documentation in audio_separator/separator/roformer/README.md for new parameters

### Manual Testing
- [x] T047 Manual test with existing BSRoformer models from integration test suite
- [x] T048 Manual test with existing MelBandRoformer models from integration test suite  
- [x] T049 Manual test with newer models containing new parameters (if available)
- [x] T050 Manual validation of error messages for common configuration issues

## Dependencies

### Critical Dependencies (TDD)
- Tests (T004-T015) MUST complete and FAIL before implementation (T016-T033)
- T001-T003 (setup) before everything else
- T016-T020 (data models) before T021-T024 (validators)
- T025-T028 (model updates) before T031-T033 (loaders)
- T029-T030 (normalization fixes) before T031-T033 (loaders)

### Implementation Dependencies
- T016 (ModelConfiguration) blocks T017, T018, T021-T024
- T020 (ParameterValidationError) blocks T021-T024
- T025-T030 (model updates) before T031-T033 (loading system)
- T031-T033 (loading system) before T034-T038 (integration)
- Implementation (T016-T038) before polish (T039-T050)

## Parallel Execution Examples

### Phase 3.2 - Contract Tests (can run simultaneously)
```
Task: "Contract test RoformerLoaderInterface.load_model in tests/contract/test_roformer_loader_interface.py"
Task: "Contract test ParameterValidatorInterface.validate_required_parameters in tests/contract/test_parameter_validator_interface.py" 
Task: "Integration test existing older model compatibility in tests/integration/test_roformer_backward_compatibility.py"
Task: "Integration test newer model with new parameters in tests/integration/test_roformer_new_parameters.py"
```

### Phase 3.3 - Data Models (can run simultaneously)
```
Task: "Implement ModelConfiguration dataclass in audio_separator/separator/roformer/model_configuration.py"
Task: "Implement BSRoformerConfig class in audio_separator/separator/roformer/bs_roformer_config.py"
Task: "Implement MelBandRoformerConfig class in audio_separator/separator/roformer/mel_band_roformer_config.py"
Task: "Implement ParameterValidationError exception in audio_separator/separator/roformer/parameter_validation_error.py"
```

### Phase 3.5 - Unit Tests (can run simultaneously)
```
Task: "Unit tests for ModelConfiguration validation in tests/unit/test_model_configuration.py"
Task: "Unit tests for ParameterValidator methods in tests/unit/test_parameter_validator.py"
Task: "Unit tests for ConfigurationNormalizer methods in tests/unit/test_configuration_normalizer.py"
Task: "Unit tests for fallback mechanism logic in tests/unit/test_fallback_loader.py"
```

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- All tests must be written first and must fail before implementation (TDD)
- Commit after each task completion
- Manual testing (T047-T050) requires actual model files
- Integration with existing test framework ensures no regression
- Fallback mechanism provides safety net for edge cases

## Validation Checklist

- [x] All contracts have corresponding tests (T004-T008)
- [x] All entities have model tasks (T016-T020)
- [x] All tests come before implementation (Phase 3.2 before 3.3)
- [x] Parallel tasks are truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] TDD ordering enforced (tests fail before implementation)
- [x] Integration scenarios from quickstart covered (T009-T013)
- [x] Audio quality validation included (T014-T015, T043)
- [x] Manual testing milestones defined (T047-T050)
