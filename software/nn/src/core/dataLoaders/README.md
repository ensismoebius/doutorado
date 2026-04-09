# dataLoaders

Purpose
- Implement dataset backends and batch sources used by experiments: MAT file loaders, SQLite-backed sources, windowing logic, and prefetching.

Key Components
- `DataLoader` / `BatchPrefetcher` — iteration and background prefetching primitives.
- `SqliteBatchSource` — DB-backed sample provider (used for reproducible/ASan-clean tests).
- `mat_file.*` — MAT file reading and helpers.

How to use
- Include the headers from `include/nn/dataLoaders`.
- Construct a `DataLoader` with a dataset object and iterate to obtain batches.

CMake
- Target: `dataLoaders`

Tests and Examples
- Unit tests are under `src/core/dataLoaders/tests/` (GTest). Use them as usage examples.

Recent updates
- `SqliteBatchSource` now pre-reserves trial accumulation buffers using per-trial SQLite blob-byte estimates before row decoding.
- Window sample assembly now reserves per-window output capacity based on active window mode (`EegWindow`, `AudioWindow`, `FusedWindow`) to reduce hot-loop allocations.
- `DataLoaderIterator::operator*()` now reuses the cached `fetch_batch()` path and no longer rebuilds per-batch index vectors redundantly.
- `SqliteBatchSource::open_db()` failure diagnostics now use `NN_LOG_ERROR` instead of direct stderr writes.
- Windowed datasets now implement type-specific `print(IDatasetPrinter&)` dispatch and `WindowingDatasetPrinter` outputs detailed modality-aware summaries (window specs, hop/overlap, per-row window factors, and feature counts) instead of only generic totals.
- Printer implementations were split from `src/core/dataLoaders/10.1117/dataset_info.cpp` into dedicated translation units: `datasets/raw/Dataset101117Printer.cpp` and `datasets/windowed/WindowingDatasetPrinter.cpp`.

Optimization techniques and references
- Trial buffer pre-reservation (`eeg_accum`, `audio_accum`): capacity planning from SQLite blob-size metadata to avoid geometric reallocation/copy churn in hot decode loops (see [1], [2]).
- Per-window `samp.reserve(...)`: branch-aware preallocation to reduce allocator pressure and improve cache locality while assembling windows (see [1], [2]).
- Iterator deduplication (`operator*` uses `fetch_batch`): remove redundant index-vector construction to reduce unnecessary per-batch work and memory traffic (see [3]).
- Unified logger channel (`NN_LOG_ERROR`): consolidate diagnostics path for deterministic output capture and lower observability drift across runtime modes (see [4]).

Bibliographic references
- [1] Ulrich Drepper. What Every Programmer Should Know About Memory. Red Hat, 2007.
- [2] Paul R. Wilson, Mark S. Johnstone, Michael Neely, and David Boles. Dynamic Storage Allocation: A Survey and Critical Review. IWMM, 1995.
- [3] Robert C. Martin. Clean Code: A Handbook of Agile Software Craftsmanship. Prentice Hall, 2008.
- [4] Martin Kleppmann. Designing Data-Intensive Applications. O'Reilly, 2017.
# DataLoader Sampling Architecture

This module uses a sampler-driven architecture inspired by large ML frameworks while keeping a small C++20 core API.

## Goals

- Keep `DataLoader` focused on batching and collation only.
- Move index selection policy to interchangeable sampler components.
- Allow deterministic epoch-aware sampling when needed.

## Core Contracts

- `Dataset` provides sample access and collation.
- `ISampler` produces index sequences, not data.
- `DataLoader` asks the sampler for one epoch of indices and forms mini-batches from those indices.

### `ISampler`

Declared in `include/nn/dataLoaders/samplers/ISampler.hpp`:

- `index_count()`: number of indices produced per epoch.
- `set_epoch(std::size_t)`: hook for stateful samplers.
- `sample_into(std::span<std::size_t>)`: fills caller-provided buffer.

Using `sample_into(std::span<...>)` avoids hidden allocations inside samplers and keeps the data path explicit.

## File Layout

- Interface: `include/nn/dataLoaders/samplers/ISampler.hpp`
- Built-in sampler headers:
	- `include/nn/dataLoaders/samplers/SequentialSampler.hpp`
	- `include/nn/dataLoaders/samplers/RandomSampler.hpp`
	- `include/nn/dataLoaders/samplers/WeightedRandomSampler.hpp`
	- `include/nn/dataLoaders/samplers/DistributedSampler.hpp`
- Implementations:
	- `src/core/dataLoaders/samplers/SequentialSampler.cpp`
	- `src/core/dataLoaders/samplers/RandomSampler.cpp`
	- `src/core/dataLoaders/samplers/WeightedRandomSampler.cpp`
	- `src/core/dataLoaders/samplers/DistributedSampler.cpp`

## Built-in Samplers

- `SequentialSampler(dataset_size)`: emits `[0, 1, ..., N-1]`.
- `RandomSampler(dataset_size, seed)`: emits a full permutation, deterministic per epoch when seeded.
- `WeightedRandomSampler(weights, num_samples, seed)`: draws with replacement using `std::discrete_distribution`.
- `DistributedSampler(dataset_size, num_replicas, rank, shuffle, drop_last, seed)`: shard-aware sampler for multi-worker training.

## DataLoader Usage

### Backward-compatible API

```cpp
DataLoader loader(dataset, batch_size, true, 42U);
```

- `do_shuffle=false` maps to `SequentialSampler`.
- `do_shuffle=true` maps to `RandomSampler(seed)`.

### Preferred sampler-injected API

```cpp
auto sampler = std::make_unique<WeightedRandomSampler>(weights, num_samples, 42U);
DataLoader loader(dataset, batch_size, std::move(sampler));
```

## Epoch Semantics

- On each `begin()` call, `DataLoader` calls `sampler->set_epoch(epoch_)` then `sample_into(...)`.
- This allows deterministic epoch-varying behavior (for example `seed + epoch`).
- Iterator independence is preserved because each iterator owns its index snapshot.

## Extension Guide

To add a custom strategy:

1. Implement `ISampler`.
2. Keep `sample_into` O(k) for `k=index_count()` and validate output span size.
3. Keep all policy/state inside the sampler.
4. Inject it through `DataLoader(std::shared_ptr<Dataset>, std::size_t, std::unique_ptr<ISampler>)`.

No `DataLoader` modification is needed for new sampling policies.

## 10.1117 module layout (dataset code organization)

The repository contains a dataset module for the 10.1117 paper. Sources and public headers were reorganized into focused subfolders to improve discoverability.

- Source tree: `src/core/dataLoaders/10.1117/`
	- `loaders/` — dataset-specific loaders (`AudioLoader.cpp`, `EEGLoader.cpp`)
	- `schema/` — metadata and discovery helpers (`SubjectDiscovery.cpp`, `METADATA` helpers)
	- `codec/` — codecs and formatters (`InputModeCodec.cpp`, `BatchTargetFormatter.cpp`)
	- `datasets/raw/` — protocol/raw dataset and raw batching helpers (`Dataset101117.cpp`, `SamplePacking.cpp`, `SynchronizedBatchAssembler.cpp`, `Dataset101117Printer.cpp`)
	- `datasets/windowed/` — windowed datasets and printer (`AudioWindowDataset.cpp`, `EEGWindowDataset.cpp`, `FusedWindowDataset.cpp`, `WindowingDatasetPrinter.cpp`)

- Public headers: `include/nn/dataLoaders/10.1117/` mirrors this layout:
	- `loaders/`, `schema/`, `codec/`, `datasets/raw/`, `datasets/windowed/`

The current public headers present in the repository include (examples):

- [include/nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp](include/nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp)
- [include/nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp](include/nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp)
- [include/nn/dataLoaders/10.1117/datasets/windowed/AudioWindowDataset.hpp](include/nn/dataLoaders/10.1117/datasets/windowed/AudioWindowDataset.hpp)
- [include/nn/dataLoaders/10.1117/datasets/windowed/EEGWindowDataset.hpp](include/nn/dataLoaders/10.1117/datasets/windowed/EEGWindowDataset.hpp)
- [include/nn/dataLoaders/10.1117/datasets/windowed/FusedWindowDataset.hpp](include/nn/dataLoaders/10.1117/datasets/windowed/FusedWindowDataset.hpp)
- [include/nn/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp](include/nn/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp)

Note: legacy `WindowedDatasetPrinter.hpp` has been removed; use `WindowingDatasetPrinter.hpp` for modality-aware summaries.

Public headers reorganization
- Recently the protocol and windowing public headers were moved under a
	`datasets/` subfolder to better group dataset variants and dataset-specific
	utilities. New public header paths include, for example:

	- `nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp`
	- `nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp`
	- `nn/dataLoaders/10.1117/datasets/windowed/AudioWindowDataset.hpp`
	- `nn/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp`

	Dataset printer classes were extracted into dedicated headers under the
	same `datasets/` tree (e.g. `Dataset101117Printer.hpp`,
	`WindowingDatasetPrinter.hpp`) and now compile from dedicated translation
	units (`datasets/raw/Dataset101117Printer.cpp` and
	`datasets/windowed/WindowingDatasetPrinter.cpp`).

Migration notes
 - The repository's sources and tests have been updated to reference the new
	`datasets/` locations (now using `raw` and `windowed` subtrees). Old
	forwarding headers were removed and the header contents moved in-place;
	if you maintain external code that included the older paths, update
	includes accordingly to `nn/dataLoaders/10.1117/datasets/raw/...` or
	`nn/dataLoaders/10.1117/datasets/windowed/...`.

If you add new dataset files, place sources under the appropriate subfolder in `src/core/dataLoaders/10.1117/` and update `src/core/dataLoaders/10.1117/CMakeLists.txt` to reference the new paths.

## Quick build & test

After changes, reconfigure and build, then run tests:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build -j4 --output-on-failure
```

All dataset-related tests live under `src/core/dataLoaders/10.1117/tests/` and should pass after moves/edits.
