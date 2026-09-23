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
public:
    // Default sampler (shuffled or sequential) built internally:
    DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size,
        bool do_shuffle = true, std::optional<unsigned int> seed = std::nullopt);
    // Default sampler with explicit options:
    DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size,
        const DefaultSamplerOptions& options);
    // Caller-supplied sampler (e.g. FoldSampler, DistributedSampler):
    DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size,
        std::unique_ptr<ISampler> sampler);

    using Iterator = DataLoaderIterator;
    auto begin() -> Iterator;
    auto end() -> Iterator;

private:
    std::shared_ptr<Dataset> dataset_;
    std::size_t batch_size_;
    std::unique_ptr<ISampler> sampler_;
    std::size_t num_batches_;
    mutable std::size_t epoch_ = 0;
};
```

### Samplers

A **sampler**'s only job is to decide, for a given epoch, the list of sample
indices to use and in what order — it never touches the actual data, only the
list of "which rows to read next":

```cpp
// File: include/data_loaders/samplers/ISampler.hpp
// Samplers produce dataset indices only (not data).
class ISampler
{
public:
    virtual ~ISampler() = default;

    // Number of indices yielded per epoch.
    [[nodiscard]] virtual auto index_count() const noexcept -> std::size_t = 0;

    // Optional epoch hook (e.g., reseed deterministic shuffles).
    virtual void set_epoch(std::size_t epoch) = 0;

    // Fill `out` with sampled indices for the current epoch.
    virtual void sample_into(std::span<std::size_t> out) = 0;
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
index; `collate_into()` is what turns several individually-fetched samples into a
single `Batch` (with a legacy `collate()` still present for older call sites):

```cpp
// File: include/data_loaders/datasets/Dataset.hpp
class Dataset
{
public:
    [[nodiscard]] virtual auto get_item(std::size_t idx) const -> Batch = 0;
    [[nodiscard]] virtual auto size() const -> std::size_t = 0;

    // Default impl: fetches each index via get_item() and packs rows into
    // `batch.inputs`/`batch.targets`, reusing `batch`'s storage when the
    // shape already matches (avoids a reallocation every call).
    virtual void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const;
};
```

## Transforms

**The problem this solves.** A raw window straight off disk is rarely what a
network should train on. Two examples from this project:

- An EEG window's absolute voltage scale depends on the amplifier and
  electrode, not the underlying brain signal — a network trained on raw
  microvolts would partly be learning "which recording session was this,"
  not "what pattern is in the signal."
- `meeting01`'s AudioMNIST loader used to slice every recording into
  non-overlapping windows and then always keep window 0 of each recording
  when a per-recording cap applied. Window 0 of nearly every AudioMNIST file
  turned out to be near-silent lead-in (raw int16 values around ±2..±12) —
  so the autoencoder was training on 32 ms of silence, not the spoken digit,
  and its near-zero reconstruction error looked like perfect learning when it
  was actually trivial memorization of one near-constant pattern. See
  [Experiments/Meeting01.md](../Experiments/Meeting01.md), "Fixed: AudioMNIST
  window degeneracy," for the full story.

Both problems have the same shape: a fixed, hard-coded transformation baked
into the loader is wrong for some inputs, but rewriting the loader itself
every time is fragile and not reusable. PyTorch's answer is
`torchvision.transforms` [76]: small, composable, callable objects — a
`Compose([RandomCrop(32), Normalize(mean, std)])` pipeline that any dataset
can apply to any sample. This project has the same idea under
`nn::transforms` (`include/utility/ITransform.hpp` + `Compose.hpp`):

```cpp
// File: include/utility/ITransform.hpp
class ITransform
{
public:
    virtual ~ITransform() = default;
    virtual auto operator()(const nn::Tensor& x) const -> nn::Tensor = 0;
};
```

Every transform is a pure function object: one tensor in, one tensor out, no
mutation of the input. `Compose` chains several of them into one:

```cpp
// File: include/utility/Compose.hpp
using namespace nn::transforms;
auto pipeline = Compose({
    std::make_shared<WindowZScore>(),
});
const nn::Tensor normalized = pipeline(raw_window);
```

### What's available

| Transform | Shape it expects | What it does | Confuse it with… |
|---|---|---|---|
| `WindowZScore` | `(N, 1)` — one channel, N samples | One mean/std over the **whole** tensor | `EEGWindowZScore` — different axis! |
| `EEGWindowZScore` | `(channels, time)` | Mean/std **per row** (per channel) | `WindowZScore` |
| `AudioMeanStdNormalize` | `(samples, features)` | Mean/std **per column**, *fitted* across a whole batch/dataset first (`accumulate()` + `finalize()`), not derived from one sample | `WindowZScore` |
| `FusedModalityTransform` | `(rows, eeg_cols + audio_cols)` | Applies one transform to the EEG column block and another to the audio column block of the same tensor | — |
| `RandomCrop` | `(N, 1)`, `N ≥ crop_size` | Returns a `crop_size`-long window at a uniformly random offset — the *continuous-signal* analogue of `torchvision.transforms.RandomCrop` | `RandomIndexCrop` |
| `RandomIndexCrop` | none (operates on `vector<size_t>` indices, not tensors) | Returns a random permutation of a set of already-materialized candidate indices | `RandomCrop` |

### The confusable pair: three normalizers, three different axes

All three normalizers compute the same formula, `(x - mean) / std`, but
disagree about *which* elements the mean and std are computed over. Applying
the wrong one is a **silent** failure — it runs without error and produces a
plausible-looking tensor, just not a meaningfully normalized one:

```
WindowZScore on a (256, 1) window:
  one mean, one std, over all 256 samples.        ✓ correct for this shape

EEGWindowZScore on that SAME (256, 1) window:
  "per row" = per one of the 256 rows, over its 1 column.
  mean of 1 element = itself → subtracting it gives 0.
  std of 1 element = 0 → every output becomes 0 (only the eps floor survives).
  → the whole window silently collapses to (near) zero. No exception, no
    warning — just a window that trains as if the network saw all-zero input.
```

`AudioMeanStdNormalize` fails differently but just as quietly if given a
single `(N, 1)` sample instead of a `(samples, features)` batch: it treats
column 0 as the only feature and computes its mean/std across `N` rows
(samples), which is *a* valid computation, just not the "normalize this one
window" operation the caller likely intended.

**Rule of thumb**: the normalizer's expected shape must match the caller's
actual shape *before* it's plugged into a `Compose` pipeline — nothing checks
this for you at the type level, because `ITransform::operator()` accepts any
`(rows, cols)` tensor.

### The confusable pair: RandomCrop vs. RandomIndexCrop

Both fix the "always the same window" bias, but they operate at different
levels because `meeting01`'s loaders pre-slice every window up front instead
of loading lazily (windows need a stable `window_id`/`source_window_index`
for leakage-safe splitting — see [Meeting01.md](../Experiments/Meeting01.md)):

```
RandomCrop:        one long, continuous signal  →  crop ONE random window from it
                    [ raw signal, 4000 samples ]
                              ↓ pick random offset, e.g. 1280
                         [window, 256 samples]

RandomIndexCrop:   several ALREADY-CUT candidate windows  →  a random ORDER to drain them in
                    [w0] [w1] [w2] [w3] [w4]   (already sliced, already have window_id)
                              ↓ shuffle
                    [w3] [w0] [w4] [w1] [w2]   (caller takes candidates off the front)
```

`meeting01`'s `stratified_window_cap` (`Meeting01Dataset.cpp`) uses
`RandomIndexCrop` for exactly this reason: it needs to pick which of a
recording's *already-computed* windows to keep, not to produce a brand-new
crop from raw audio. `RandomCrop` is the more general, literal
`torchvision`-style transform — useful for any future dataset that loads
lazily from a continuous signal instead of pre-slicing.

### Determinism

Both `RandomCrop` and `RandomIndexCrop` own a private `std::mt19937` seeded
once at construction and **advance it on every call** — this matches
`torchvision.transforms.RandomCrop`'s stateful-callable behaviour: you seed
once, then call the object repeatedly, and reproducibility comes from that
one seed, not from re-seeding before each call. Two instances built with the
same seed and called the same number of times produce byte-identical output;
the same instance called twice in a row does *not* (that's the point — it's
what breaks the "always index/offset 0" bias).

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

// Create dataset from MAT file (global namespace, not nn::data_loaders;
// eagerly loads both named variables and validates matching row counts)
auto dataset = std::make_shared<MatFileDataset>("data.mat", "inputs", "targets");

// Create data loader with random shuffling
DataLoader loader(dataset, /*batch_size=*/32, /*do_shuffle=*/true, /*seed=*/42U);

// Iterate batches
for (const auto& batch : loader)
{
    // batch is a Batch{inputs, targets}; batch.inputs has shape (32, features)
    auto output = model.forward(batch.inputs, true);
    // ... training step
}
```

### K-Fold example

```cpp
// File: include/data_loaders/samplers/FoldSampler.hpp
#include "data_loaders/samplers/FoldSampler.hpp"
#include "statistics/kfold.hpp"

// Compute the fold split first (see Concepts/K-Fold-Cross-Validation.md)
statistics::KFold kf(5, /*shuffle=*/true, /*random_seed=*/42U);
statistics::FoldSplit split = kf.split(dataset->size())[fold_index];

// One FoldSampler per partition, wrapped into a DataLoader via the
// unique_ptr<ISampler> constructor overload:
auto train_sampler = std::make_unique<FoldSampler>(split, FoldPartition::Train);
auto val_sampler   = std::make_unique<FoldSampler>(split, FoldPartition::Validation);

DataLoader train_loader(dataset, batch_size, std::move(train_sampler));
DataLoader val_loader(dataset, batch_size, std::move(val_sampler));
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

// Session objects (filePath may be a .mat file or a .sqlite database;
// subject_id scopes queries when reading from sqlite, -1 = no scope):
nn::dataLoaders::AudioSession audio_session(db_path, subject_id);
nn::dataLoaders::EEGSession eeg_session(db_path, subject_id);

auto [audio_tensor, stimulus, eeg_index] = audio_session.readRow(row_index);
auto [eeg_tensor, labels] = eeg_session.readRow(row_index); // labels = {modality, stimulus, artifact}

// Or the stateless MAT-file convenience functions:
auto [audio, stim, eeg_idx] = nn::dataLoaders::loadAudioFromMat(mat_path, row_index);
auto [eeg, eeg_labels]      = nn::dataLoaders::loadEEGFromMat(mat_path, row_index);
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

4. **Applying a normalizer with the wrong shape convention.** `WindowZScore`
   (whole-tensor), `EEGWindowZScore` (per-row), and `AudioMeanStdNormalize`
   (per-column, fitted) all silently "succeed" on a tensor of the wrong
   shape — see the Transforms section above for exactly how each one
   degenerates.

5. **Not reshuffling between epochs.** If the same order is used every epoch,
   the network can start to memorise the sequence of batches rather than
   learning general patterns from the data itself.

## See Also

- [Tensor](./Tensor.md) — the data structure batches are packaged into
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — the full cross-validation story
- [Data Normalisation](../Concepts/Data-Normalisation.md) — preprocessing inputs before they reach the network
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — the EEG dataset's scientific context
- [Research-Context](../Research-Context.md) — how this data fits into the thesis overall
- [Experiments/Meeting01.md](../Experiments/Meeting01.md) — the AudioMNIST window degeneracy bug that motivated `RandomCrop`/`RandomIndexCrop`

## References

> In-text numbers follow the project-wide numbering in [References](../References.md). The entries cited above are reproduced here.

[6] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in Proc. 14th Int. Joint Conf. Artificial Intelligence (IJCAI), 1995, pp. 1137–1143.

[76] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019. [Online]. Available: https://arxiv.org/abs/1912.01703
