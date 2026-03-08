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
- `set_epoch(std::size_t)`: optional epoch hook for stateful samplers.
- `sample_into(std::span<std::size_t>)`: fills caller-provided buffer.

Using `sample_into(std::span<...>)` avoids hidden allocations inside samplers and keeps the data path explicit.

## File Layout

- Interface: `include/nn/dataLoaders/samplers/ISampler.hpp`
- Umbrella include: `include/nn/dataLoaders/Sampler.hpp`
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
