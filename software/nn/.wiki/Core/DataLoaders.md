# DataLoaders

Training data rarely lives in the format a network needs — it typically starts
as files on disk (WAV audio, `.mat` files, SQLite databases) and has to be
read, grouped into batches, and shuffled before it can be fed to the network.
This page covers the pieces of `nn` that handle that: `Dataset` (how to read
one sample), `ISampler` (which samples to use, and in what order), and
`DataLoader` (the object you actually iterate over during training).

## Theoretical Background

A data loader's job has four parts:

1. **Batching** — grouping many individual samples into one "batch" tensor,
   because training on one sample at a time is both slow (poor use of
   hardware parallelism) and noisy (a single sample's gradient is a bad
   estimate of the true direction to move the weights). This is why training
   is usually called **stochastic** gradient descent: each step uses a random
   subset (the batch), not the whole dataset.
2. **Shuffling** — presenting samples in a different random order every
   epoch, so the network can't accidentally learn to exploit whatever order
   the samples happen to be stored in (for example, if the file listing
   happens to group all of one class together).
3. **Prefetching** — reading and preparing the *next* batch while the
   network is still busy computing on the *current* one, so time spent
   loading data from disk overlaps with compute time instead of adding to it.
4. **Sampling strategy** — deciding not just the order but *which* samples
   go into training vs. validation in a given run (see k-fold below).

### K-Fold Cross-Validation

A single train/validation split can be misleading — a model might just get
lucky (or unlucky) with which samples ended up in which split. **K-fold
cross-validation** guards against this by splitting the dataset into $k$
equal-sized chunks ("folds") and running $k$ separate train/evaluate rounds,
each time holding out a *different* fold as validation and training on the
rest [6]. The final result is usually the average performance across all $k$
rounds, which is far less sensitive to any one lucky/unlucky split:

```
Fold 1: [val] [train train train]
Fold 2: [train] [val train train]
Fold 3: [train train] [val train]
Fold 4: [train train train] [val]
```

See [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) for the
full explanation, including the "nested" variant used when you also need to
tune hyperparameters without leaking information from the test set.

## How It Is Implemented Here

```cpp
// File: include/data_loaders/runtime/DataLoader.hpp
class DataLoader
{
    std::shared_ptr<Dataset> dataset_;
    std::size_t batch_size_;
    std::unique_ptr<ISampler> sampler_;

public:
    using Iterator = DataLoaderIterator;

    auto begin() -> Iterator;
    auto end() -> Iterator;
};
```

### Samplers

A **sampler**'s only job is to decide, for a given epoch, the list of sample
indices to use and in what order — it never touches the actual data, only the
list of "which rows to read next":

```cpp
// File: include/data_loaders/samplers/ISampler.hpp
class ISampler
{
public:
    virtual auto get_indices(size_t epoch) -> std::vector<size_t> = 0;
};
```

Available samplers:
- `SequentialSampler` — always the same order, no shuffling. Useful mainly
  for debugging, where you want a reproducible, inspectable sequence.
- `RandomSampler` — shuffles using a seeded random number generator, so the
  shuffle is different every epoch but the whole run is still reproducible
  given the same seed.
- `FoldSampler` — implements the k-fold train/validation split described
  above.
- `DistributedSampler` — splits the dataset across multiple GPUs/processes so
  each one trains on a different slice.

### Dataset interface

A **dataset** knows how many samples it has and how to fetch one of them by
index; `collate()` is what turns several individually-fetched samples into a
single batch tensor:

```cpp
// File: include/data_loaders/datasets/Dataset.hpp
class Dataset
{
public:
    virtual auto size() const -> size_t = 0;
    virtual auto get(size_t index) -> Tensor = 0;
    virtual auto collate(const std::vector<size_t>& indices) -> Tensor = 0;
};
```

## Data Flow

```mermaid
flowchart TB
    subgraph Data
        Raw[Raw Files<br/>.mat, .csv]
    end

    subgraph Loading
        DS[Dataset]
        SP[Sampler]
        DL[DataLoader]
    end

    subgraph Batching
        Collate[collate()]
        Prefetch[BatchPrefetcher]
    end

    subgraph Training
        Batch[Batch Tensor]
    end

    Raw --> DS
    DS --> SP
    SP --> DL
    DL --> Collate
    Collate --> Prefetch
    Prefetch --> Batch
```

## Usage Example

```cpp
// File: include/data_loaders/runtime/DataLoader.hpp
#include "data_loaders/runtime/DataLoader.hpp"
#include "data_loaders/datasets/MatFileDataset.hpp"

// Create dataset from MAT file
auto dataset = std::make_shared<nn::data_loaders::MatFileDataset>("data.mat");

// Create data loader with random shuffling
DataLoader loader(dataset, batch_size=32, do_shuffle=true, seed=42);

// Iterate batches
for (const auto& batch : loader)
{
    // batch is a Tensor of shape (32, features)
    auto output = model.forward(batch, true);
    // ... training step
}
```

### K-Fold example

```cpp
// File: include/data_loaders/samplers/FoldSampler.hpp
#include "data_loaders/samplers/FoldSampler.hpp"

// Create k-fold sampler
nn::data_loaders::FoldSampler fold_sampler(
    dataset->size(),  // total samples
    5,                // number of folds
    fold_index,       // which fold is validation
    seed
);

// Train on fold
DataLoader train_loader(dataset, batch_size, fold_sampler.get_train_indices());
DataLoader val_loader(dataset, batch_size, fold_sampler.get_val_indices());
```

## Domain-Specific Loaders

### 10.1117/12.2255697 — EEG imagined-speech dataset (thesis dataset)

This is the public dataset the thesis validates against: 15 Spanish-speaking
subjects, recorded saying vowels and directional commands under three
conditions ("modalities") — spoken aloud, imagined silently, or a mix of the
two.

```
include/data_loaders/10.1117/
  schema/
    Metadata.hpp        # dataset-wide metadata constants
    Names.hpp           # speaker/command name tables
  loaders/
    AudioLoader.hpp     # loads phonated speech WAVs
    EEGLoader.hpp       # loads raw EEG channel data
    AudioData.hpp       # AudioData value type
    EEGData.hpp         # EEGData value type
```

```cpp
#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"

auto audio = nn::dataLoaders::AudioLoader::load(root, speaker, command);
auto eeg   = nn::dataLoaders::EEGLoader::load(root, speaker, command);
```

The EEG channels most relevant to imagined speech are F7 and T5 (near
Wernicke's area, associated with language comprehension) and Fp1, F3, F7 (near
Broca's area, associated with speech production). See
[Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) for the
neuroscience background on why these regions matter.

#### Float32 blob detection (AudioLoader + EEGLoader)

A practical gotcha worth knowing about if you touch this loader: the source
SQLite database stores each audio recording as 4-byte-per-sample (`float32`)
binary blobs, not the 8-byte-per-sample (`float64`) you might assume from a
quick glance at typical scientific data. Both loaders check the *actual* byte
count of each blob against both possible sizes and read it accordingly,
rather than assuming one or the other:

```cpp
const size_t n = ImaginedSpeechSchema_10_1117.audioSamples();
const size_t expected_float  = n * sizeof(float);
const size_t expected_double = n * sizeof(double);
float* dst = audioSamples.mutable_data_ptr();
if (static_cast<size_t>(bytes) == expected_float)
{
    const float* src = reinterpret_cast<const float*>(blob);
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}
else if (static_cast<size_t>(bytes) == expected_double)
{
    const double* src = reinterpret_cast<const double*>(blob);
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
}
else
    throw std::runtime_error("AudioLoader(SQL): unexpected audio blob size");
```

`EEGLoader.cpp` follows the same pattern. Without this check, loading the
original database throws `"unexpected audio blob size"`, because a real blob
is 705,600 bytes — which is $176{,}400 \times 4$ (float32), not
$176{,}400 \times 8$ (float64) as you might otherwise expect.

See [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) for the
neuroscience context and [Research-Context](../Research-Context.md) for how
this dataset fits into the thesis as a whole.

---

## Common Pitfalls

1. **Batch size too large or too small.** Too large, and the model tends to
   generalise worse (it sees fewer, "smoother" gradient estimates over
   training); too small, and training becomes slow and the gradient estimates
   become noisy.

2. **Forgetting to set a seed.** Without a fixed random seed, shuffling and
   sampling are different every run, which makes results impossible to
   reproduce — always set one explicitly for experiments you intend to report.

3. **Not prefetching on GPU training.** Without `BatchPrefetcher`, the GPU
   sits idle while the next batch is being read from disk and assembled; with
   it, that reading happens in parallel with the current batch's computation.

4. **Not reshuffling between epochs.** If the same order is used every epoch,
   the network can start to memorise the sequence of batches rather than
   learning general patterns from the data itself.

## See Also

- [Tensor](./Tensor.md) — the data structure batches are packaged into
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — the full cross-validation story
- [Data Normalisation](../Concepts/Data-Normalisation.md) — preprocessing inputs before they reach the network
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — the EEG dataset's scientific context
- [Research-Context](../Research-Context.md) — how this data fits into the thesis overall

## References

[1] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in *Proc. 14th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 1995, pp. 1137–1143.

[2] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019. [Online]. Available: https://arxiv.org/abs/1912.01703

> In-text numbers follow the project-wide numbering in [References](../References.md). The entries cited above are reproduced here.

[6] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in Proc. 14th Int. Joint Conf. Artificial Intelligence (IJCAI), 1995, pp. 1137–1143.
