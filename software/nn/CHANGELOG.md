# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Added sampler abstraction in `include/nn/dataLoaders/Sampler.hpp` with built-in implementations:
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
