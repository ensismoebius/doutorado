# Samplers

Purpose
- Sampling strategies used by `DataLoader` such as `SequentialSampler`, `RandomSampler`, `WeightedRandomSampler`, and `DistributedSampler`.

Usage
- Create a sampler and pass it to your dataset/DataLoader configuration to control iteration order.

Key classes
- `SequentialSampler`, `RandomSampler`, `WeightedRandomSampler`, `DistributedSampler`.

Tests
- See `src/core/dataLoaders/samplers/tests/` for examples and expected semantics.
