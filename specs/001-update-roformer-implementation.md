# Feature Specification: Update Roformer Implementation

**Feature Branch**: `001-update-roformer-implementation`  
**Created**: September 25, 2025  
**Status**: Draft  
**Input**: User description: "update Roformer implementation: this audio-separator project currently has an older implementation of the Roformer architecture inference code, copied from another project over a year ago into folder path audio_separator/separator/uvr_lib_v5/ this works well for many models, but some of the latest Roformer models don't work with the older inference code; the model fails to load with errors such as "AttributeError: "'norm'" - File "/Users/andrew/miniforge3/lib/python3.13/site-packages/audio_separator/separator/uvr_lib_v5/tfc_tdf_v3.py", line 155, in __init__  norm = get_norm(norm_type=config.model.norm)" or " TypeError: BSRoformer.init() got an unexpected keyword argument 'mlp_expansion_factor'". I've copied the latest inference code from the other project into this folder path, as a reference: audio_separator/separator/msst-models-new we need to identify the differences between the old and new roformer implementations and modify the audio-separator implementation to work with the newest models without breaking support for older ones. It's critical that we don't break existing functionality, so we should be careful to understand the old and new code fully before making changes, and get me to manually test to validate things still work whenever we've made changes."

## Execution Flow (main)
```
1. Parse user description from Input
   → Identified: Need to update Roformer implementation for latest model compatibility
2. Extract key concepts from description
   → Actors: developers, users with existing models, users with new models
   → Actions: update implementation, maintain backward compatibility, validate functionality
   → Data: Roformer models (old and new), inference code, model configurations
   → Constraints: cannot break existing functionality, must support both old and new models
3. For each unclear aspect:
   → [RESOLVED] Clear requirements provided with specific error messages and paths
4. Fill User Scenarios & Testing section
   → Clear user flow: load models, separate audio, validate outputs
5. Generate Functional Requirements
   → All requirements are testable and specific
6. Identify Key Entities
   → BSRoformer, MelBandRoformer, model configurations, inference parameters
7. Run Review Checklist
   → No clarifications needed, no implementation details included
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
Users need to be able to load and use the latest Roformer models for audio separation without losing the ability to use their existing older models. The system should seamlessly handle both old and new model formats, providing the same quality audio separation results.

### Acceptance Scenarios
1. **Given** a user has an existing older Roformer model, **When** they attempt to separate audio, **Then** the separation should work exactly as it did before the update
2. **Given** a user has a newer Roformer model with updated parameters (mlp_expansion_factor, sage_attention, zero_dc), **When** they attempt to load and use the model, **Then** the model should load successfully and produce high-quality audio separation
3. **Given** a user has models that use different normalization configurations, **When** they load these models, **Then** the system should handle the normalization appropriately without AttributeError exceptions
4. **Given** a user switches between old and new model types, **When** they perform multiple separations in sequence, **Then** all separations should complete successfully without conflicts

### Edge Cases
- What happens when a model configuration contains parameters that exist in new implementation but not old?
- How does system handle models with missing or invalid configuration parameters?
- What happens when a user tries to load a corrupted or incompatible model file?
- How does the system behave when switching between different Roformer variants (BSRoformer vs MelBandRoformer)?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST load and execute older Roformer models without any regression in functionality or performance, preferentially using new inference code with fallback to old code if loading fails
- **FR-002**: System MUST load and execute newer Roformer models that include updated parameters (mlp_expansion_factor, sage_attention, zero_dc, use_torch_checkpoint, skip_connection) using the new inference code
- **FR-003**: System MUST handle both BSRoformer and MelBandRoformer model variants, attempting new implementation first and falling back to old implementation only if necessary
- **FR-004**: System MUST gracefully handle model configurations with missing required parameters by failing with detailed error messages that specify which parameters are missing and their expected types/values
- **FR-005**: System MUST resolve normalization configuration issues that cause AttributeError exceptions in tfc_tdf_v3.py
- **FR-006**: System MUST maintain identical audio separation quality for existing models after the update, validated using spectral analysis comparison with waveform and spectrogram similarity thresholds (≥0.90 for waveform, ≥0.80 for spectrogram)
- **FR-007**: System MUST provide clear error messages when model loading fails due to incompatible configurations, including specific parameter mismatches and suggested corrections
- **FR-008**: System MUST support backward compatibility for all existing model files and configurations
- **FR-009**: System MUST validate model configurations before attempting to instantiate models
- **FR-010**: System MUST allow seamless switching between different Roformer model types within the same session
- **FR-011**: System MUST undergo manual testing validation after completing major implementation milestones (e.g., new inference code integration, backward compatibility implementation, error handling updates)

### Key Entities *(include if feature involves data)*
- **Roformer Model**: Audio separation model with specific architecture parameters, exists in old and new variants with different parameter sets
- **Model Configuration**: Dictionary containing model parameters including dimension, depth, attention settings, and normalization preferences
- **BSRoformer**: Band-split Roformer variant that processes audio in frequency bands, requires freqs_per_bands parameter
- **MelBandRoformer**: Mel-scale band Roformer variant that uses mel-scale frequency bands, requires num_bands parameter
- **Model Parameters**: Configuration values including mlp_expansion_factor, sage_attention, zero_dc, and other architecture-specific settings

## Clarifications

### Session 2025-09-25
- Q: The spec mentions "identical audio separation quality" must be maintained, but doesn't specify how quality should be measured or what tolerance is acceptable for validation testing. → A: Use spectral analysis comparison with defined similarity thresholds
- Q: How should the system determine appropriate defaults for missing model parameters? → A: Fail gracefully with detailed error messages rather than assuming defaults
- Q: How should the system detect whether a model uses the old or new Roformer format? → A: All Roformer models should use new inference code (test first), fallback to old if fails
- Q: When should manual testing validation be performed during the implementation process? → A: Test only after completing major implementation milestones

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous  
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---
