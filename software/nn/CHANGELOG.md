# CHANGELOG

All notable changes to this project will be documented in this file.

## Unreleased (2026-05-02)

### Changed

- Trainer progress callbacks now expose fractional intra-batch progress via `TrainingState::batch_progress`, and Experiment04 batch bars advance while the active batch is still being assembled and executed instead of only jumping at batch completion.
- Tensor frontend decoupling: `include/nn/tensor/Tensor.hpp` no longer depends on concrete backend headers (`XTensorBackend`/`OpenCLTensorBackend`) and no longer provides a concrete-backend default template argument.
- Active tensor alias now resolves through central backend selection (`nn::Backend`), preserving single-point backend binding in `include/nn/Backend.hpp`.
- OpenCL runtime verification API now takes `OpenCLTensorBackend` inputs directly, removing dependency on global `nn::Tensor` alias for backend-local verification probes.
- Added backend switchability coverage (`tensor_backend_switchability_gtest`) with a custom backend conformance test and source-contract assertions that guard `Tensor.hpp` against concrete-backend leakage.
- Added CMake backend-reference enforcement (`cmake/BackendImplementationGuard.cmake`): build now generates a guard translation unit that emits a compile-time error with the exact message `Backend implementation must only be refereced inside include/nn/Backend.hpp !` when concrete backend tokens are used outside allowed boundaries.

- OpenCL linear algebra: added a tiled local-memory kernel for direct
  `A^T * B` computation (`matmul_lhs_transposed_kernel`) and integrated it
  through `OpenCLTensorBackend::matmul_lhs_transposed(...)`.
- Linear backward (`dL/dW`) on OpenCL now uses the tuned lhs-transposed
  primitive instead of materializing transpose + matmul in the hot path.
- Benchmark coverage was extended to include explicit comparison between
  the new lhs-transposed path and the prior transpose-based route.

### Validation

- Added/updated targeted correctness coverage for lhs-transposed matmul in
  OpenCL backend tests.
- Long-run benchmark samples (`NN_TENSOR_BENCH_ITERS=20`) showed stable
  speedup for grad-weight computation on the active AMD rusticl stack:
  roughly 1.76x to 1.83x faster than the transpose-based probe.

## Unreleased (2026-03-17)

- Refactor: Move experiment pipeline into `LstmAutoencoderExperiment` class
  - Integrated the LSTM autoencoder runner into `autoencoderRunner`; new API in
    `src/experiments/autoencoderRunner/lib/include/guayaquil.hpp` and
    `src/experiments/autoencoderRunner/lib/src/guayaquil.cpp`.
  - Folded former `src/experiments/04` library, tests, and profiles into
    `src/experiments/autoencoderRunner/lib`, `src/experiments/autoencoderRunner/tests`, and
    `src/experiments/autoencoderRunner/profiles`.

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

### Changed (2026-04-21)

- Experiment04 runner behavior unified: `src/experiments/guayaquil/lib/src/guayaquil.cpp`
  now defaults to the comparative SNN-vs-LSTM pipeline.
- Legacy standalone LSTM execution path was removed from Experiment04 runner;
  execution now always delegates to the comparative SNN-vs-LSTM pipeline.
- Legacy CLI aliases (`--guayaquil`, `--lstm-autoencoder`,
  `--lstm-profile`, `--config`) were removed. The runner now accepts only
  comparative CLI flags (for example, `--comparative` and
  `--comparative-config`).

### Fixed (2026-04-16)

- SNN stability: `Leaky` and `LeakyIntegrator` now clamp effective membrane
  parameters (`R`, `C`) to positive minima during dynamics/gradient evaluation so
  `tau = R*C` and `beta = exp(-dt/tau)` remain numerically stable when optimizers
  push raw parameters toward non-positive values.
- `LeakyBPTT` backward consistency:
  - readout mode now computes resistance/capacitance gradients from the same
    non-spiking recurrence used in forward mode (no spike/reset reconstruction path).
  - threshold gradient now includes explicit direct and recurrent reset-path terms
    instead of the previous simplified accumulator.
  - capacitance is now a trainable parameter (`Tensor`) exposed by `params()` and
    updated in backward pass (aligned with `Leaky`/`LeakyIntegrator`).
- Surrogate safety: `ExponentialSurrogate` and `BoxcarSurrogate` now validate
  constructor hyperparameters and throw `std::invalid_argument` for non-positive
  `sharpness` / `window` to prevent divide-by-zero and degenerate gradients.

### Changed (2026-04-03)

- OpenCL backend initialization checks are now backend-owned via
  `OpenCLTensorBackend::initialize_runtime_or_throw(bool)`. Experiment entry
  points no longer perform sanitizer/runtime availability checks directly.
- OpenCL runtime activity verification (GPU busy probe + reconstruction-MSE
  workload) is now backend-owned via
  `OpenCLTensorBackend::verify_runtime_activity_or_throw(...)`, reducing
  OpenCL-specific code in `autoencoderRunner`.
- OpenCL runtime init/shutdown lifecycle is now backend-owned through
  `OpenCLTensorBackend::RuntimeScope` and
  `OpenCLTensorBackend::start_runtime_scope_or_throw(...)`.

### Added

- Added integrated Experiment04 LSTM assets under the AutoencoderRunner module:
  `src/experiments/autoencoderRunner/lib/include/guayaquil`,
  `src/experiments/autoencoderRunner/lib/src/guayaquil`, `src/experiments/autoencoderRunner/tests`, and
  `src/experiments/autoencoderRunner/profiles/lstm-*.json`.

- Dual-branch multimodal autoencoders for `autoencoderRunner`: separate EEG and audio encoder branches
  with latent-space fusion and modality-specific decoders. Implementations live under
  `src/experiments/autoencoderRunner/lib/` (Fused & Protocol autoencoders).
  - Trainable SNN membrane parameters: membrane resistance `R` and capacitance `C` are now
  learnable scalar parameters exposed via profile fields (`ae.resistance`, `ae.capacitance`) and
  persisted with model weights.
- Transparent architecture fallback: models can automatically fall back from `DualBranchFusion`
  to `ResidualDense` when modality split hints are missing.
- Shared builder utilities: `AutoencoderBuilders.hpp` centralizes encoder/decoder construction
  helpers used across ANN and SNN autoencoders.
- `reset_state()` support and proper membrane state clearing added to SNN modules (Leaky layers
  and Sequential wrappers).
- Redesign test suite: added `src/experiments/autoencoderRunner/tests/AutoencoderRedesign_gtest.cpp` with tests
  covering dual-branch and dense fallback modes for ANN and SNN.

### Changed

- `autoencoderRunner` model construction now infers protocol/eeg/audio splits and auto-selects
  `DualBranchFusion` when applicable. See `src/experiments/autoencoderRunner/lib/src/autoencoderRunner.cpp`.
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
- Recommended reviewer focus: `src/experiments/autoencoderRunner/lib/*`, `include/nn/layers/Leaky.hpp`, and
  `src/experiments/autoencoderRunner/tests/AutoencoderRedesign_gtest.cpp` for behavioral review.

