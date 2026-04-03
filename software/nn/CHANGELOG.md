# CHANGELOG

All notable changes to this project will be documented in this file.

## Unreleased (2026-03-17)

- Refactor: Move experiment pipeline into `Experiment04` class
  - `src/experiments/03/experiment04.cpp` refactored; new API in
    `src/experiments/03/lib/include/experiment04.hpp` and
    `src/experiments/03/lib/src/experiment04.cpp`.

- Prefetcher: `BatchPrefetcher` redesigned to single-producer thread with
  a bounded deque to serialize MAT I/O and avoid concurrent `matio` reads.

- Dataset summary: added `printDatasetSummary()` with a fast `AudioWithEEG`
  estimate (uses `min(audio_rows, eeg_rows)` by default); optional
  `--exact-summary` can compute exact counts at higher I/O cost.

- Progress UI: added in-place single-line progress helper (`progress.hpp/.cpp`)
  with effective-total logic and display capping.

- Formatting: updated `.clang-format` to allow more aggressive breaking of
  function return types and parameter lists; applied `clang-format` across
  tracked `.cpp`/`.hpp` files.

- Documentation: updated `.github/copilot-instructions.md` to reflect
  the refactors, threading and I/O constraints, and repository conventions.

### Notes
- Public header changes and API-breaking changes require a changelog entry
  and a migration guide; consider bumping the project's semantic version.
- Recommend reviewing and committing formatting-only changes as a separate
  commit to simplify review of behavioral changes.
## Unreleased (2026-03-20)

### Changed (2026-04-03)

- OpenCL backend initialization checks are now backend-owned via
  `OpenCLTensorBackend::initialize_runtime_or_throw(bool)`. Experiment entry
  points no longer perform sanitizer/runtime availability checks directly.
- OpenCL runtime activity verification (GPU busy probe + reconstruction-MSE
  workload) is now backend-owned via
  `OpenCLTensorBackend::verify_runtime_activity_or_throw(...)`, reducing
  OpenCL-specific code in `experiment03`.
- OpenCL runtime init/shutdown lifecycle is now backend-owned through
  `OpenCLTensorBackend::RuntimeScope` and
  `OpenCLTensorBackend::start_runtime_scope_or_throw(...)`.

### Added

- Dual-branch multimodal autoencoders for `experiment03`: separate EEG and audio encoder branches
  with latent-space fusion and modality-specific decoders. Implementations live under
  `src/experiments/03/lib/` (Fused & Protocol autoencoders).
- Trainable SNN membrane parameters: membrane resistance `R` and capacitance `C` are now
  learnable scalar parameters exposed via the CLI (`--ae-resistance`, `--ae-capacitance`) and
  persisted with model weights.
- Transparent architecture fallback: models can automatically fall back from `DualBranchFusion`
  to `ResidualDense` when modality split hints are missing.
- Shared builder utilities: `AutoencoderBuilders.hpp` centralizes encoder/decoder construction
  helpers used across ANN and SNN autoencoders.
- `reset_state()` support and proper membrane state clearing added to SNN modules (Leaky layers
  and Sequential wrappers).
- Redesign test suite: added `src/experiments/03/tests/AutoencoderRedesign_gtest.cpp` with tests
  covering dual-branch and dense fallback modes for ANN and SNN.

### Changed

- `experiment03` model construction now infers protocol/eeg/audio splits and auto-selects
  `DualBranchFusion` when applicable. See `src/experiments/03/lib/src/experiment03.cpp`.
- Refactored `ProtocolAutoencoder` and `ProtocolSpikingAutoencoder` to support both
  dual-branch and dense fallback execution paths.

- I/O and config formats: runtime ingestion of YAML and `.npz` (NumPy) files has been
  disabled in this branch in favor of JSON configuration. The project now vendors
  `nlohmann::json` and expects experiment configs as `.json` files (e.g. `spec.json`,
  `config.json`). The `cnpy` library remains vendored for tooling and future use,
  but runtime NPZ loading paths are guarded or disabled to avoid silent runtime
  dependencies on `.npz` artifacts.

### Fixed

- Implemented `Leaky::reset_state()` to correctly zero internal membrane state, resolving
  state-leak issues across batches and enabling deterministic reset behavior in SNN tests.

### Notes

- Validation: focused regression and redesign suites passing locally (redesign tests + Leaky
  and serializer regression slice). Static analysis (cppcheck, flawfinder) reported no new
  issues related to these changes; existing findings are in unrelated data-loader tests.
- Recommended reviewer focus: `src/experiments/03/lib/*`, `include/nn/layers/Leaky.hpp`, and
  `src/experiments/03/tests/AutoencoderRedesign_gtest.cpp` for behavioral review.

