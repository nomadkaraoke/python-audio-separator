<!--
Sync Impact Report:
Version change: Initial → 1.0.0
Modified principles: All (new constitution)
Added sections: Core Principles, Performance Standards, Quality Assurance
Removed sections: None (template placeholders)
Templates requiring updates:
- ✅ plan-template.md (Constitution Check section updated)
- ✅ spec-template.md (aligned with principles)
- ✅ tasks-template.md (TDD enforcement aligned)
- ✅ agent-file-template.md (no changes needed)
Follow-up TODOs: None
-->

# Audio Separator Constitution

## Core Principles

### I. Library-First Architecture
The `Separator` class MUST be the primary interface for all audio separation functionality. CLI and remote API are thin wrappers around the library. Libraries MUST be self-contained, independently testable, and documented with clear separation of concerns between architectures (MDX, VR, Demucs, MDXC).

**Rationale**: This ensures the core functionality can be integrated into other projects while maintaining a consistent API across all interfaces.

### II. Multi-Interface Consistency
Every core feature MUST be accessible via three interfaces: Python API, CLI, and Remote API. Parameter names and behavior MUST be identical across all interfaces. All interfaces MUST support the same model architectures and processing options.

**Rationale**: Users should have consistent experience regardless of how they access the functionality, enabling seamless transition between local and remote processing.

### III. Test-First Development (NON-NEGOTIABLE)
TDD is mandatory: Tests written → Tests fail → Implementation → Tests pass. All new features MUST include unit tests, integration tests with audio validation (SSIM comparison), and CLI tests. No code merges without passing tests.

**Rationale**: Audio processing requires precision and consistency. Automated testing with perceptual validation ensures output quality remains stable across changes.

### IV. Performance & Resource Efficiency
Hardware acceleration MUST be supported (CUDA, CoreML, DirectML). Memory usage MUST be optimized for large audio files through streaming and batch processing. Processing parameters MUST be tunable for different hardware capabilities.

**Rationale**: Audio separation is computationally intensive. Efficient resource usage enables processing of longer files and broader hardware compatibility.

### V. Model Architecture Separation
Each model architecture (MDX, VR, Demucs, MDXC) MUST be implemented in separate modules inheriting from `CommonSeparator`. Loading one architecture MUST NOT load code from others. Architecture-specific parameters MUST be isolated and documented.

**Rationale**: This prevents conflicts between different model types and keeps memory usage minimal by loading only required components.

## Performance Standards

All audio processing operations MUST meet these requirements:
- **Memory efficiency**: Support files larger than available RAM through streaming
- **GPU utilization**: Automatically detect and utilize available hardware acceleration
- **Batch processing**: Support processing multiple files without model reloading
- **Output consistency**: Identical inputs MUST produce identical outputs (deterministic)

## Quality Assurance

### Testing Requirements
- **Unit tests**: All core classes and functions
- **Integration tests**: End-to-end audio processing with SSIM validation
- **Performance tests**: Memory usage and processing speed benchmarks
- **Cross-platform tests**: Windows, macOS, Linux compatibility

### Audio Validation
Output quality MUST be validated using:
- Waveform and spectrogram image comparison (SSIM ≥ 0.95)
- Reference audio files for each supported model architecture
- Automated regression testing on model output changes

## Governance

This constitution supersedes all other development practices. All pull requests MUST verify compliance with these principles. Any deviation MUST be explicitly justified and documented.

**Amendment Process**: Changes require documentation of impact, approval from maintainers, and migration plan for affected code.

**Compliance Review**: All features undergo constitutional compliance check during planning phase and post-implementation validation.

**Version**: 1.0.0 | **Ratified**: 2025-09-25 | **Last Amended**: 2025-09-25