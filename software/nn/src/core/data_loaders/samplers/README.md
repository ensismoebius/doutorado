# samplers

The sampling policies `DataLoader` asks for each epoch's indices: `SequentialSampler`,
`RandomSampler`, `FoldSampler`, `WeightedRandomSampler`, `DistributedSampler`. Headers in
[`include/data_loaders/samplers/`](../../../../include/data_loaders/samplers/); tests in `tests/`.

How they fit together, and how to add one:
[`../README.md` › Sampling architecture](../README.md#sampling-architecture).
