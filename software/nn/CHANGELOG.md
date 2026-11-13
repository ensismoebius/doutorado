# CHANGELOG

All notable changes to this project will be documented in this file.

## 2026-11-13 — static analysis at zero

### Fixed

- `.clang-tidy` ran 131 of the intended ~400 checks: its `Checks` string was a folded block with
  `#` comments inside, which YAML keeps as text, so every glob after a comment matched nothing
  and only `clang-analyzer-*` ran. Rewritten (comments outside the string); `WarningsAsErrors: "*"`.
- Defects the full check set found:
  - the tensor comma initializer (`Tensor t; t << 1, 2, 3;`) silently accepted too few or too
    many values; it now throws `std::invalid_argument` (and stays quiet during unwinding);
  - `countShardRows` (`mat_file_utils.cpp`) ran a shell python command through `popen` and read a
    missing `count` as 0; it now parses the JSON with nlohmann and requires each count;
  - `read_csv_signal` read an empty or non-numeric field as 0; it now throws;
  - `generate_autoencoder_spike_data` drew from a global `random_device` engine; it takes a seed;
  - `NetworkSerializer::loadParametersInto` swallowed a missing LIF parameter (`try {} catch {}`)
    and kept the constructed value: a different network from the one saved. Now required;
  - meeting01: `extract_latent_size` fell back to 16 and the layer-width parsers swallowed a bad
    width; both throw, naming the key to set;
  - `reduce_time_major_output` tested `time_steps <= 0` after a cast to unsigned (never true);
  - the SQLite test helper ignored a failing step; `analysis-all` passed when a tool was absent.
- `thesis_output_gtest` wrote `e05_output_test/` into the source tree; it now uses a temp dir.

### Removed

- meeting01's LaTeX/pgfplots exports (`write_latex_exports`, the `.dat` writers, five CSV writers
  that wrote 0.0 for a missing family) and their tests: nothing read them.
  `dataset.latex_data_dir` must now be empty.
- `scripts/ci/validate_static_analysis.py`, the unread `.cppcheck` file and the never-run
  `software/nn/.github/workflows/` (GitHub runs only the repository-root workflows).

### Changed

- `analysis-cppcheck` reads `compile_commands.json` (each file with its real include paths) and
  fails on any finding. `useStlAlgorithm` and `shadowFunction` are off on purpose; false
  positives carry an inline `cppcheck-suppress` with the reason. See
  `.wiki/Guides/Static-Analysis.md`.

## 2026-11-13 — legacy experiments removed

### Removed

- The legacy experiments `autoencoderRunner` (was 03, with its grid scripts and profiles),
  `waveletAE` (was 02) and `paraconsistentBaseline` (was 00), the shared `experiments_config`
  library they used, `scripts/data/e03_dedup_master_table.py`, and their wiki pages
  (`Experiments/AutoencoderRunner.md`, `WaveletAE.md`, `ParaconsistentBaseline.md`,
  `Guides/Grid-Runbook.md`). None of them feeds the thesis or the paper; git history keeps them.

### Changed

- `meeting01_lib` is defined in `src/experiments/meeting01/CMakeLists.txt` (it was inside the
  autoencoderRunner build file) and links the core libraries directly. `thesis_lib`, the
  `nn_microscope` bindings and the autoencoder demo link `autoencoder_models` instead of
  `autoencoderRunner_lib`.
- The tests of core models that lived with autoencoderRunner moved to the core:
  `src/core/models/autoencoder/tests/autoencoder_models_gtest.cpp` (suite
  `AutoencoderModelsTest`, was `AutoencoderRunnerRedesignTest`; it guards the thesis SNN-AE
  firing-rate regularization) and `src/core/models/lstm/tests/lstm_autoencoder_training_gtest.cpp`.
- The PGO example workload is the thesis `debug.json` profile.

## 2026-11-13 — documentation overhaul

### Fixed

- `waveletAE` no longer falls back to synthetic samples when its data cannot be read (its
  hard-coded placeholder paths meant it always did), and no longer skips a wavelet that
  yields no window: both are errors. The data paths are the required spec keys
  `data.eeg_path` / `data.audio_path` (`src/experiments/waveletAE/spec.example.json`); the
  spec path is a required argument.
- `waveletAE` and `paraconsistentBaseline` printed nothing, errors included: their
  `StreamRedirector` suppresses the Logger's console output until a progress printer drains
  it, and neither has one. Removed.
- `run_thesis_profiles.sh`: new `biomex` scope; `start-over-all` runs phase00 → phase01 →
  biomex; `all` no longer includes the BIOMEX-DB profiles (sorted first, they ran before
  Phase 00); with no argument in a pipe it refuses instead of defaulting to `all`.
  `run_thesis_smoke.sh` gains `biomex`.
- `progress_gtest` is registered with `gtest_discover_tests` like every other suite (it was
  one opaque `add_test`).
- Removed two dangling symlinks (`src/protocols/`) and the unused thesis draft `backup.tex`.

### Documentation

- New entry points: repository `README.md`, `software/nn/README.md`, `documentation/README.md`,
  `notebooks/README.md`, `src/experiments/paraconsistentGA/README.md`; new wiki guides
  `How-To-Change-Anything`, `Data-Bases`, `Thesis-Document`; every `src/core/*/README.md`
  rewritten from the tree; `scripts/README.md` and the thesis experiment README rewritten.
- Stale paths (`include/nn/…`, `docs/…`, deleted scripts), a merged citation (Guido 2019 was
  given the Coretto DOI), the non-existent `nn_gtest_discover_tests` wrapper, and the
  hand-edited `Layers.hpp` instructions were corrected.
- CI runs the five Meeting01 pytest suites.

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
    `src/experiments/autoencoderRunner/lib/include/meeting01.hpp` and
    `src/experiments/autoencoderRunner/lib/src/meeting01.cpp`.
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

- Experiment04 runner behavior unified: `src/experiments/meeting01/lib/src/meeting01.cpp`
  now defaults to the comparative SNN-vs-LSTM pipeline.
- Legacy standalone LSTM execution path was removed from Experiment04 runner;
  execution now always delegates to the comparative SNN-vs-LSTM pipeline.
- Legacy CLI aliases (`--meeting01`, `--lstm-autoencoder`,
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
  `src/experiments/autoencoderRunner/lib/include/meeting01`,
  `src/experiments/autoencoderRunner/lib/src/meeting01`, `src/experiments/autoencoderRunner/tests`, and
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

