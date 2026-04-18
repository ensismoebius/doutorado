# Experiment03 and Core Libraries: Full Description

## Scope and intent
This document consolidates the operational description of Experiment03 and the core libraries under src/core. It focuses on architecture, execution flow, contracts, and integration points used by experiments and demos.

Primary source anchors:
- [src/experiments/03/README.md](src/experiments/03/README.md)
- [src/experiments/03/CMakeLists.txt](src/experiments/03/CMakeLists.txt)
- [src/experiments/03/experiment03.cpp](src/experiments/03/experiment03.cpp)
- [src/experiments/03/lib/include/Experiment03Config.hpp](src/experiments/03/lib/include/Experiment03Config.hpp)
- [src/experiments/03/lib/src/experiment03.cpp](src/experiments/03/lib/src/experiment03.cpp)
- [src/core/README.md](src/core/README.md)
- [src/core/CMakeLists.txt](src/core/CMakeLists.txt)

## Experiment03

### What Experiment03 is
Experiment03 is a profile-driven training runner for ANN and SNN autoencoders over the 10.1117 imagined-speech EEG+audio dataset. It supports protocol-level and sliding-window datasets, with optional k-fold cross-validation and grid-run automation.

Key executable and library targets:
- Executable: experiment03 from [src/experiments/03/CMakeLists.txt](src/experiments/03/CMakeLists.txt)
- Reusable library target: experiment03_lib from [src/experiments/03/CMakeLists.txt](src/experiments/03/CMakeLists.txt)

Core dependencies linked into Experiment03:
- util
- dataLoaders
- dataLoaders_10_1117
- dataLoaders_10_1117_windowing
- statistics
- CLI11
- matioCpp
- Eigen
- OpenMP

### Runtime entry and startup sequence
The launcher in [src/experiments/03/experiment03.cpp](src/experiments/03/experiment03.cpp) is intentionally thin and performs this order:
1. Load default profile into Config.
2. Parse CLI using default profile as baseline.
3. Configure logger level from NN_EXPERIMENT03_LOG_LEVEL.
4. Redirect streams through logging.
5. Construct Experiment03 and run.

The main orchestration lives in [src/experiments/03/lib/src/experiment03.cpp](src/experiments/03/lib/src/experiment03.cpp).

### Configuration model
The canonical runtime schema is the Config struct in [src/experiments/03/lib/include/Experiment03Config.hpp](src/experiments/03/lib/include/Experiment03Config.hpp).

Major groups:
- Program/device: program device token and OpenCL profiling switch.
- Dataset routing: subject regex, dataset root path, dataset type, protocol input mode.
- DataLoader/sampler: batch size, max batches per epoch, sampler mode, weighted/distributed controls, seeds.
- Autoencoder architecture: model family, depth, hidden/latent sizes, explicit layer specs, branch/fusion dimensions, SNN constants.
- Training policy: optimizer token and hyperparameters, loss token, epoch count, LR-plateau scheduler controls.
- Validation diagnostics: optional modality-specific validation reporting.
- Windowing: separate EEG and audio window specs.
- Prefetch controls: lookahead and RAM cap.
- K-fold controls: enable flag, split count, shuffle and seed.

Inference helpers inside Config:
- Effective input/eeg/audio feature resolution.
- Auto architecture resolution when set to Auto.

### Profile system and strictness
Profiles are loaded by [src/experiments/03/lib/src/ProfileLoader.cpp](src/experiments/03/lib/src/ProfileLoader.cpp) via API in [src/experiments/03/lib/include/ProfileLoader.hpp](src/experiments/03/lib/include/ProfileLoader.hpp).

Behavior:
- Accepts profile stem names or explicit paths.
- Searches source and working-directory candidate locations.
- Rejects unknown top-level keys.
- Allows _comment* keys for inline profile documentation.
- Parses deterministic controls such as sampler_shuffle_seed and kfold_seed.

CLI contract in [src/experiments/03/lib/src/cli.cpp](src/experiments/03/lib/src/cli.cpp):
- Public runtime flag is profile selection.
- Enforces RNG policy:
  - sampler_shuffle_seed required by policy.
  - kfold_seed required when kfold shuffle is enabled.
- Resolves sampler options through data loader sampler option resolver.

### Dataset and model compatibility matrix
Enums and helpers:
- Dataset type: [src/experiments/03/lib/include/Experiment03DatasetType.hpp](src/experiments/03/lib/include/Experiment03DatasetType.hpp)
- Autoencoder type: [src/experiments/03/lib/include/Experiment03AutoencoderType.hpp](src/experiments/03/lib/include/Experiment03AutoencoderType.hpp)

Compatibility is explicit and enforced by helper logic:
- protocol <-> protocol-ann/protocol-snn
- eeg-window <-> eeg-window-ann/eeg-window-snn
- audio-window <-> audio-window-ann/audio-window-snn
- fused-window <-> fused-window-ann/fused-window-snn

### Dataset construction path
Dataset instantiation is centralized in [src/experiments/03/lib/src/DatasetBuilder.cpp](src/experiments/03/lib/src/DatasetBuilder.cpp) and declared in [src/experiments/03/lib/include/DatasetBuilder.hpp](src/experiments/03/lib/include/DatasetBuilder.hpp).

Build routing:
- Protocol -> Dataset101117
- EegWindow -> EEGWindowDataset
- AudioWindow -> AudioWindowDataset
- FusedWindow -> FusedWindowDataset

This keeps experiment driver logic independent from dataset concrete types.

### Autoencoder construction model
Autoencoder grammars and builders are in [src/experiments/03/lib/include/AutoencoderBuilders.hpp](src/experiments/03/lib/include/AutoencoderBuilders.hpp) and concrete model files under [src/experiments/03/lib/src/autoencoder](src/experiments/03/lib/src/autoencoder).

Supported declarative layer grammar includes:
- linear:width
- linear:width:activation
- activation-only tokens
- residual and residual:N

Supported width tokens include integer widths and symbolic tokens like latent/output/branch_hidden/fusion_hidden.

Supported activation families:
- ANN path: relu, leaky_relu, identity
- SNN path: leaky, leaky_integrator, identity

Architectures supported by Protocol/Fused implementations:
- ResidualDense
- DualBranchFusion

### Training and validation pipeline
Training flow in [src/experiments/03/lib/src/experiment03.cpp](src/experiments/03/lib/src/experiment03.cpp):
1. Discover subjects and build dataset.
2. Build DataLoader once per logical train loader configuration.
3. Build model according to dataset/model family and architecture policy.
4. Create optimizer via OptimizerFactory.
5. Fit input transform for normalization policy.
6. For each epoch:
   - Create BatchPrefetcher with configured max batches/lookahead/RAM cap.
   - Iterate prefetched batches.
   - Forward -> loss -> backward -> optimizer step.
   - Aggregate epoch train loss.
7. Run validation pass with same batching protocol (no optimizer step).
8. Optionally log modality diagnostics (EEG/audio validation losses).
9. Optionally apply ReduceLROnPlateau behavior.

K-fold mode adds fold selection and per-fold train/val loaders, then aggregates fold metrics and grand mean validation loss.

### Results and reproducibility artifacts
Summary model and writing:
- Summary struct and writer API: [src/experiments/03/lib/include/ResultsWriter.hpp](src/experiments/03/lib/include/ResultsWriter.hpp)
- Summary assembly: [src/experiments/03/lib/src/RunSummaryBuilder.cpp](src/experiments/03/lib/src/RunSummaryBuilder.cpp)
- JSON persistence: [src/experiments/03/lib/src/ResultsWriter.cpp](src/experiments/03/lib/src/ResultsWriter.cpp)

Written result includes:
- profile metadata and model/dataset types
- optimizer and loss metadata
- processed samples and batches
- epoch train losses
- k-fold arrays for validation losses (overall/eeg/audio)
- fold means and grand mean
- exit code and error message

### Grid execution and analysis tooling
Operational scripts in [src/experiments/03/scripts](src/experiments/03/scripts):
- [src/experiments/03/scripts/create_test_profiles.py](src/experiments/03/scripts/create_test_profiles.py)
- [src/experiments/03/scripts/run_full_grid_and_analyze.sh](src/experiments/03/scripts/run_full_grid_and_analyze.sh)
- [src/experiments/03/scripts/analyze_grid_results.py](src/experiments/03/scripts/analyze_grid_results.py)

Grid orchestration script does:
1. Regenerate profile grid.
2. Run profiles in parallel with GNU parallel and per-job timeout.
3. Store joblog and completion marker.
4. Run post-analysis that exports comparison CSVs.

### Experiment03 test coverage map
Primary tests under [src/experiments/03/tests](src/experiments/03/tests):
- [src/experiments/03/tests/ProfileAndResults_gtest.cpp](src/experiments/03/tests/ProfileAndResults_gtest.cpp): profile parsing, key validation, result behavior.
- [src/experiments/03/tests/DatasetBuilder_gtest.cpp](src/experiments/03/tests/DatasetBuilder_gtest.cpp): dataset builder routing basics.
- [src/experiments/03/tests/AutoencoderRedesign_gtest.cpp](src/experiments/03/tests/AutoencoderRedesign_gtest.cpp): ANN/SNN architecture behavior, forward/backward shape checks, state reset semantics.

## Core libraries (src/core)

### Top-level composition
The core layer is assembled by [src/core/CMakeLists.txt](src/core/CMakeLists.txt), which adds these submodules:
- wave
- saver
- layers
- tensor
- wavelet
- utility
- statistics
- optimizers
- dataLoaders
- initializers
- linearAlgebra
- paraconsistent
- tools

The module overview starts in [src/core/README.md](src/core/README.md).

### dataLoaders
Anchor docs:
- [src/core/dataLoaders/README.md](src/core/dataLoaders/README.md)
- [src/core/dataLoaders/10.1117/README.md](src/core/dataLoaders/10.1117/README.md)
- [src/core/dataLoaders/samplers/README.md](src/core/dataLoaders/samplers/README.md)

CMake targets:
- dataLoaders
- dataLoaders_samplers
- dataLoaders_10_1117
- dataLoaders_10_1117_windowing

Responsibilities:
- Generic Dataset/DataLoader abstractions.
- Sampler-driven index policies (sequential, random, weighted, distributed, fold).
- BatchPrefetcher for bounded asynchronous producer-consumer loading.
- 10.1117-specific loaders, schema utilities, codecs, protocol datasets, and window datasets.
- SQLite-backed fast paths and deterministic fold-aware trial filtering.

Key invariants:
- DataLoader iterator semantics preserve epoch-index snapshots.
- MAT I/O reads are serialized through prefetching/loader design.

### tensor
Anchor docs:
- [src/core/tensor/README.md](src/core/tensor/README.md)

CMake target:
- tensor

Responsibilities:
- nn::Tensor abstraction used across layers and optimizers.
- Eigen backend default behavior.
- OpenCL backend path with runtime fallback and sanitizer-aware guards.
- Backend-level vectorized primitives such as rowwise bias add.

Integration:
- Core dependency for layers, optimization, and model computation.

### layers
Anchor docs:
- [src/core/layers/README.md](src/core/layers/README.md)

CMake target:
- layers

Responsibilities:
- Neural layer implementations and composition utilities.
- Dense, activation, residual, and SNN-compatible layer components consumed by autoencoders.
- Performance-sensitive forward path optimizations such as vectorized bias epilogues.

### optimizers
Anchor docs:
- [src/core/optimizers/README.md](src/core/optimizers/README.md)

CMake target:
- optimizers (interface)

Responsibilities:
- Optimizer API and concrete strategies (adam/sgd conventions).
- state_dict/load_state_dict support for checkpoint compatibility.
- OptimizerFactory for token-based runtime construction.

### statistics
Anchor docs:
- [src/core/statistics/README.md](src/core/statistics/README.md)

CMake target:
- statistics

Responsibilities:
- Metrics, confusion-matrix helpers, and classification stats.
- Deterministic KFold and StratifiedKFold splitters.
- Validation and fold tooling used directly by Experiment03.

### utility
Anchor docs:
- [src/core/utility/README.md](src/core/utility/README.md)

CMake target:
- util

Responsibilities:
- Cross-cutting helpers for batching, progress display, and supporting runtime utilities.
- Performance-focused helpers used by experiment loops.

### linearAlgebra
Anchor docs:
- [src/core/linearAlgebra/README.md](src/core/linearAlgebra/README.md)

CMake target:
- linearAlgebra

Responsibilities:
- Lower-level linear algebra bridge helpers used by tensor/layer internals.

### initializers
Anchor docs:
- [src/core/initializers/README.md](src/core/initializers/README.md)

CMake target:
- initializers (interface)

Responsibilities:
- Parameter initialization policies (xavier/kaiming and related helpers).
- Used at model/layer construction time.

### wave
Anchor docs:
- [src/core/wave/README.md](src/core/wave/README.md)

CMake target:
- waveCoreLib

Responsibilities:
- Audio/signal helpers and WAV-related utilities used by demos and preprocessing.

### wavelet
Anchor docs:
- [src/core/wavelet/README.md](src/core/wavelet/README.md)

CMake target:
- wavelet

Responsibilities:
- Wavelet transforms and decomposition utilities for time-frequency analysis.

### paraconsistent
Anchor docs:
- [src/core/paraconsistent/README.md](src/core/paraconsistent/README.md)

CMake target:
- paraconsistent

Responsibilities:
- Paraconsistent logic utilities and metrics for project-specific analysis workflows.

### saver
Anchor docs:
- [src/core/saver/README.md](src/core/saver/README.md)

CMake target:
- saver (interface)

Responsibilities:
- High-level state persistence wrappers around serialization/state_dict flows.

### tools
Anchor docs:
- [src/core/tools/README.md](src/core/tools/README.md)

Responsibilities:
- Developer tools and benchmarks (for example prefetch benchmarks) that support engineering workflows.

## How Experiment03 uses core libraries end-to-end
At runtime, Experiment03 composes core modules in this sequence:
1. statistics and dataLoaders provide fold splits, discovery, dataset objects, and batch iteration.
2. tensor and layers provide the model computation graph and forward/backward operations.
3. initializers and optimizers configure and update parameters.
4. utility/progress and logging provide observability.
5. ResultsWriter produces stable JSON run artifacts used by analysis scripts.

This composition keeps dataset policy, model architecture, optimization, and reporting modular while preserving deterministic profile-driven behavior.

## Practical extension points
Recommended extension points when adding new behavior:
- Add new profile keys and validation in [src/experiments/03/lib/src/ProfileLoader.cpp](src/experiments/03/lib/src/ProfileLoader.cpp).
- Add new dataset variant through DatasetBuilder routing and dataset implementation under dataLoaders.
- Add new model family by extending autoencoder builders and concrete autoencoder classes under [src/experiments/03/lib/src/autoencoder](src/experiments/03/lib/src/autoencoder).
- Add new optimizer token in OptimizerFactory and wire through Config parsing.
- Add regression tests under [src/experiments/03/tests](src/experiments/03/tests) and relevant src/core module tests.
