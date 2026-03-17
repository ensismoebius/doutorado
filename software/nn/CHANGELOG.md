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
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Added sampler abstraction in `include/nn/dataLoaders/samplers/ISampler.hpp` with built-in implementations:
  - `SequentialSampler`
  - `RandomSampler`
  - `WeightedRandomSampler`
  - `DistributedSampler`
- Added sampler implementation file `src/core/dataLoaders/Sampler.cpp`.
- Added sampler-focused tests in `src/core/dataLoaders/tests/sampler_gtest.cpp`.
- Added sampler architecture guide in `src/core/dataLoaders/README.md`.

### Changed

- Refactored `DataLoader` to delegate sample-index generation to `ISampler`.
- Added constructor overload for explicit sampler injection:
  - `DataLoader(std::shared_ptr<Dataset>, std::size_t, std::unique_ptr<ISampler>)`
- Kept backward-compatible constructor (`do_shuffle`, `seed`) by mapping internally to default samplers.
