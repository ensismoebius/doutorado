# data_loaders

Datasets, samplers and the `DataLoader` that batches them; MAT-file and SQLite readers; the dataset-specific loaders, each in a folder named after the dataset's DOI: `10.1117/` is the Coretto imagined-speech base (audio + EEG), `10.5281/zenodo.1342401/` is the Free Spoken Digit Dataset.

**Full documentation:** [`.wiki/Core/DataLoaders.md`](../../../.wiki/Core/DataLoaders.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/data_loaders`](../../../include/data_loaders), [`include/windowing`](../../../include/windowing) |
| Implementation (`.cpp`) | 29 file(s) in this folder |
| CMake library target | `MatTestUtilsLib`, `dataLoaders_10_1117`, `dataLoaders_10_1117_windowing`, `dataLoaders_samplers`, `data_loaders`, `data_loaders_fsdd` |
| Test executables | `dataLoaders_10_1117_gtest`, `dataLoaders_gtest`, `dataLoaders_samplers_gtest`, `fsdd_loader_gtest`, `windowing_gtest` |

## Header map

- `include/data_loaders/10.1117/codec/`: `BatchTargetFormatter.hpp`, `InputModeCodec.hpp`
- `include/data_loaders/10.1117/datasets/raw/`: `Dataset101117.hpp`, `Dataset101117Printer.hpp`, `RowRequest.hpp`, `SamplePacking.hpp`, `SynchronizedBatchAssembler.hpp`
- `include/data_loaders/10.1117/datasets/windowed/`: `AudioWindowDataset.hpp`, `EEGWindowDataset.hpp`, `FusedWindowDataset.hpp`, `WindowingDatasetPrinter.hpp`
- `include/data_loaders/10.1117/loaders/`: `AudioData.hpp`, `AudioLoader.hpp`, `EEGData.hpp`, `EEGLoader.hpp`
- `include/data_loaders/10.1117/schema/`: `Metadata.hpp`, `Names.hpp`, `SchemaIndexing.hpp`, `SqliteDataset.hpp`, `SubjectDiscovery.hpp`
- `include/data_loaders/10.5281/zenodo.1342401/datasets/`: `FsddWindowDataset.hpp`
- `include/data_loaders/10.5281/zenodo.1342401/loaders/`: `FsddLoader.hpp`, `FsddSample.hpp`
- `include/data_loaders/10.5281/zenodo.1342401/schema/`: `Metadata.hpp`, `Names.hpp`
- `include/data_loaders/datasets/`: `Dataset.hpp`, `MatFileDataset.hpp`, `TensorDataset.hpp`
- `include/data_loaders/interfaces/`: `IBatchSource.hpp`, `IDatasetPrinter.hpp`, `IMatLoader.hpp`
- `include/data_loaders/mat_io/`: `MatFileFlags.hpp`, `mat_file.hpp`, `mat_file_utils.hpp`
- `include/data_loaders/options/`: `SamplerOptionResolution.hpp`, `ShardIndex.hpp`
- `include/data_loaders/runtime/`: `BatchPrefetcher.hpp`, `DataLoader.hpp`, `DataLoaderIterator.hpp`, `DefaultSamplerType.hpp`
- `include/data_loaders/samplers/`: `DistributedSampler.hpp`, `FoldSampler.hpp`, `ISampler.hpp`, `RandomSampler.hpp`, `SequentialSampler.hpp`, `SubsetSampler.hpp`, `WeightedRandomSampler.hpp`
- `include/data_loaders/sources/`: `SqliteBatchSource.hpp`
- `include/windowing/`: `WindowInfo.hpp`, `WindowSpec.hpp`, `WindowingEngine.hpp`

## Sampling architecture

`DataLoader` only batches and collates. *Which* indices an epoch visits is the job of a sampler,
so a new sampling policy never touches `DataLoader`.

```
Dataset ──(size)──► ISampler ──(indices for one epoch)──► DataLoader ──► batches
                    set_epoch(e), sample_into(span)
```

| Sampler | Emits |
|---|---|
| `SequentialSampler(n)` | `0, 1, ..., n-1` |
| `RandomSampler(n, seed)` | a permutation, deterministic per (seed, epoch) |
| `FoldSampler(split, partition)` | the train or validation indices of one `statistics::FoldSplit` |
| `WeightedRandomSampler(weights, k, seed)` | `k` draws with replacement |
| `DistributedSampler(n, replicas, rank, shuffle, drop_last, seed)` | one shard per worker |

```cpp
DataLoader loader(dataset, batch_size, /*shuffle=*/true, 42U);          // Sequential/Random
DataLoader loader(dataset, batch_size, std::make_unique<FoldSampler>(split, part));
```

On every `begin()` the loader calls `set_epoch(epoch)` and then `sample_into(...)`, so a seeded
sampler is reproducible and still changes order between epochs. To add a policy: implement
`ISampler` (headers in [`include/data_loaders/samplers/`](../../../include/data_loaders/samplers/)),
keep all state inside it, and inject it through the constructor above. Sampler sources and tests: [`samplers/`](samplers/README.md).

## Adding a new dataset

1. Put its loader in a folder named after the dataset's DOI, as `10.1117/` and
   `10.5281/zenodo.1342401/` are, with the same subfolders (`loaders/`, `schema/`, `datasets/`).
   Headers go under `include/data_loaders/<doi>/`, sources under `src/core/data_loaders/<doi>/`.
2. Give it its own `CMakeLists.txt` and add it with `add_subdirectory` in this folder's
   `CMakeLists.txt`; add a `tests/` folder registered with `gtest_discover_tests(<exe>)`.
3. Document it in [`.wiki/Core/DataLoaders.md`](../../../.wiki/Core/DataLoaders.md) and, if an
   experiment uses it, in that experiment's wiki page.

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target dataLoaders_10_1117_gtest
  ctest --test-dir out/build/max-performance -R 'BatchTargetFormatterTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R dataLoaders_10_1117_gtest` silently matches nothing.

  Suites here: `BatchTargetFormatterTest`, `InputModeCodecTest`, `MockDatasetGeneratorTest`, `AudioLoaderArticleSpecTest`, `AudioLoaderTest`, `AudioLoaderStandaloneTest`, `EEGLoaderArticleSpecTest`, `EEGLoaderEdgeTest`, `EEGLoaderTest`, `SqliteSession`, `SqliteDatasetOpen`, `Dataset101117ModesTest` (and 33 more).
- After a behaviour change, update [`.wiki/Core/DataLoaders.md`](../../../.wiki/Core/DataLoaders.md) (and `.wiki/References.md` if a paper is involved).
