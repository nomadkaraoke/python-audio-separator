### Post-Implementation Issues: Roformer Routing and Execution Paths

- **Observed behavior**: New Roformer models load and separate successfully, but execution is routed through `MDXCSeparator` rather than a dedicated Roformer separator. Logs show the MDXC path with Roformer-specific branching inside `MDXCSeparator`.

- **Root cause**:
  - `Separator.list_supported_model_files()` groups Roformer entries under the `MDXC` model type, so `Separator.load_model()` instantiates `architectures.mdxc_separator.MDXCSeparator` for Roformer files.
  - `separator_classes` in `separator.py` has no mapping for a `Roformer` type; the dedicated `architectures/roformer_separator.py` is never routed to and is effectively unused.
  - `MDXCSeparator` contains an internal Roformer branch (constructs `BSRoformer`/`MelBandRoformer`, loads checkpoints, and implements Roformer chunking/overlap-add). This duplicates responsibility with the new `roformer_loader`.

- **Impact**:
  - The new Roformer loader (normalization/validation/fallback) is not used in actual runs; loader stats remain zero in logs.
  - Two overlapping Roformer code paths (inside `MDXCSeparator` and the new loader + `RoformerSeparator`) create confusion and increase maintenance cost.
  - `RoformerSeparator` is dead code under current routing.

- **Decision**: Proceed with Option A (minimal-change refactor)
  - Keep routing via `MDXCSeparator` to avoid broad changes.
  - Refactor the Roformer branch in `MDXCSeparator` to use `RoformerLoader` for model loading/validation/fallback, while preserving the existing Roformer chunking/overlap-add execution.
  - Align the Roformer detection flag naming (`is_roformer_model`) with `CommonSeparator`, and surface loader stats via existing `CommonSeparator.get_roformer_loading_stats()` and `Separator` logging.

- **Planned edits (Option A)**:
  - `architectures/mdxc_separator.py`:
    - Use `self.is_roformer_model` (from `CommonSeparator`) instead of a separate `self.is_roformer` detection.
    - In `load_model()`, when Roformer is detected, call `self.roformer_loader.load_model(model_path=self.model_path, config=self.model_data['model'], device=str(self.torch_device))`. On success, set `self.model_run = result.model`; on failure, fall back to existing direct instantiation path.
    - Keep current Roformer chunking/overlap-add logic in `demix()` unchanged.
  - `roformer/roformer_loader.py`:
    - Ensure it returns a `ModelLoadingResult` that matches the local dataclass (use `success_result`, `failure_result`, `fallback_success_result` helpers), and populate metadata via `model_info` rather than custom fields.

- **Expected outcomes**:
  - No change in separation outputs or performance characteristics.
  - Loader stats in logs reflect actual usage (non-zero `new_implementation_success` and/or `fallback_success`).
  - Reduced duplication and clearer ownership, while deferring larger routing changes (introducing `RoformerSeparator`) to a follow-up if desired.


