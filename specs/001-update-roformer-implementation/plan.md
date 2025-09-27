# Implementation Plan: Update Roformer Implementation

**Branch**: `001-update-roformer-implementation` | **Date**: September 25, 2025 | **Spec**: [/specs/001-update-roformer-implementation.md](../001-update-roformer-implementation.md)
**Input**: Feature specification from `/specs/001-update-roformer-implementation.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Update the Roformer implementation to support latest model architectures while maintaining backward compatibility. The primary requirement is enabling newer Roformer models with updated parameters (mlp_expansion_factor, sage_attention, zero_dc) while preserving functionality for existing older models. Technical approach involves integrating new inference code with fallback mechanisms and comprehensive validation using existing spectral analysis testing framework.

## Technical Context
**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch, librosa, soundfile, numpy, onnxruntime  
**Storage**: Model files (.ckpt, .pth, .onnx), audio files (FLAC, WAV, MP3)  
**Testing**: pytest with custom audio validation (SSIM comparison)  
**Target Platform**: Cross-platform (Windows, macOS, Linux) with GPU acceleration support  
**Project Type**: single - Python library with CLI wrapper  
**Performance Goals**: Maintain existing separation quality (≥0.90 waveform, ≥0.80 spectrogram similarity)  
**Constraints**: Zero regression in existing model functionality, backward compatibility mandatory  
**Scale/Scope**: Support for BSRoformer and MelBandRoformer variants, integration with existing 4 model architectures

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Library-First Architecture
- [x] Core functionality implemented in `Separator` class or similar library pattern
- [x] CLI/Remote API are thin wrappers, not containing business logic
- [x] Clear separation between model architectures (MDX, VR, Demucs, MDXC)

### II. Multi-Interface Consistency  
- [x] Feature accessible via Python API, CLI, and Remote API (if applicable)
- [x] Parameter names identical across all interfaces
- [x] Same model architectures supported across interfaces

### III. Test-First Development (NON-NEGOTIABLE)
- [x] Tests written before implementation
- [x] Unit tests for all core functionality
- [x] Integration tests with audio validation (SSIM comparison)
- [x] CLI tests for all exposed functionality

### IV. Performance & Resource Efficiency
- [x] Hardware acceleration support considered (CUDA, CoreML, DirectML)
- [x] Memory optimization for large files (streaming/batch processing)
- [x] Tunable parameters for different hardware capabilities

### V. Model Architecture Separation
- [x] Each architecture in separate modules
- [x] Inherits from `CommonSeparator` pattern
- [x] Architecture-specific parameters isolated
- [x] Loading one architecture doesn't load others

## Project Structure

### Documentation (this feature)
```
specs/001-update-roformer-implementation/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
audio_separator/
├── separator/
│   ├── uvr_lib_v5/          # Existing old Roformer implementation
│   ├── msst-models-new/     # New reference implementation
│   ├── common_separator.py  # Base class for all architectures
│   └── roformer_separator.py # Updated Roformer implementation
└── utils/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: Option 1 - Single project structure, as this is a library enhancement rather than web/mobile application

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - Analyze differences between old and new Roformer implementations
   - Research parameter compatibility between BSRoformer and MelBandRoformer variants
   - Investigate normalization configuration handling patterns
   - Study fallback mechanism implementation strategies

2. **Generate and dispatch research agents**:
   ```
   Task: "Research differences between old uvr_lib_v5 and new msst-models Roformer implementations"
   Task: "Find best practices for backward compatibility in PyTorch model loading"
   Task: "Research parameter validation patterns for ML model configurations"
   Task: "Study fallback mechanism implementations in audio processing libraries"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical unknowns resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - RoformerModel: Configuration parameters, validation rules
   - ModelConfiguration: Parameter dictionaries, normalization settings
   - BSRoformerConfig: Band-split specific parameters
   - MelBandRoformerConfig: Mel-scale band specific parameters

2. **Generate API contracts** from functional requirements:
   - Model loading interface with fallback capability
   - Parameter validation interface
   - Error reporting interface for configuration mismatches
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - Test model loading with old format
   - Test model loading with new format
   - Test fallback mechanism activation
   - Test parameter validation and error reporting

4. **Extract test scenarios** from user stories:
   - Load existing older Roformer model → separation works identically
   - Load newer Roformer model → separation works with new parameters
   - Switch between model types → no conflicts or failures

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh cursor`
   - Add Roformer implementation context
   - Preserve existing audio separation knowledge
   - Update with new model parameter handling

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No constitutional violations identified - all checks passed*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
