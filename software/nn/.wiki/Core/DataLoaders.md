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
| `bandpass_notch` (`nn::utility`, not `ITransform`) | `vector<float>`, one **full recording**, not a window | FIR bandpass + optional mains-notch — see its own subsection below for why it can't be a window-level `ITransform` | — |
| `make_activity_mask` (`meeting01`, not `ITransform`) | takes `(valid_length, window_size)` ints, returns `(window_size, 1)` | 1.0/0.0 mask marking a window's zero-padded tail — see its own subsection below | `RandomIndexCrop` — different problem, see that box |
| `GaussianNoise` | `(N, 1)`, any N | Elementwise `out = x + N(0, std²)`; `std=0` is an exact no-op — see its own subsection below | `WindowZScore` — this adds, that normalizes |

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

### `bandpass_notch` — not an `ITransform` (recording-level, not window-level)

`nn::utility::bandpass_notch` (`include/utility/BandpassNotchFilter.hpp`) removes
sub-0.5 Hz drift, above-40 Hz muscle/EMG noise, and mains hum (50 Hz Siena/Italy,
60 Hz eegmmidb/US) from a **full, un-windowed** EEG recording before
`EegWindowDataset` slices it into windows. It deliberately does *not* implement
`ITransform`: a sharp low-frequency cutoff needs hundreds of FIR taps for a
usable transition width, which would dwarf a 256-sample window — filtering has
to happen once on the whole recording, upstream of windowing, the same
"continuous signal, not a discrete already-sliced candidate" territory
`RandomCrop` lives in (see the confusable-pair box above), just at the
preprocessing stage instead of the sampling stage.

**A real bug found while building this, left unfixed on purpose**: this
project already had FIR filter *coefficient generators*
(`include/wave/filter_operations.hpp` — `createLowPassFilter`,
`createStopBandFilter`, `bandStopFilter`). Their windowed-sinc kernel is
min-max-rescaled to `[0, 1]` after generation
(`buildSincLowPassKernel`, `filter_operations.cpp`). A sinc lowpass kernel
needs negative side-lobe taps to cancel stopband frequencies — squashing every
tap into `[0, 1]` removes all of them, so the result is not a working filter,
just something sinc-*shaped*. The existing tests for these functions only
check that the output matches the function's *own* (buggy) formula — never an
actual frequency response — so this passed CI silently. Nothing in the
codebase currently calls these functions outside their own tests (confirmed
via `find_references`), so the blast radius today is zero, but do **not**
reuse them for real filtering without fixing the normalization first (unity
DC-gain, i.e. divide by `sum(taps)`, not min-max). `bandpass_notch` is a
from-scratch, independently-tested implementation — it does not depend on or
share code with the broken one.

### Activity mask — excluding zero-padding from the loss, not from the window

**The problem.** `FsddWindowDataset` (used by both `fsdd` and `audiomnist`) slices each
recording into non-overlapping `window_size`-sample windows. When a recording's length
isn't a multiple of `window_size`, the *last* window is shorter, and the loader fills the
rest with zeros so every window still has a fixed shape. Those zeros were never recorded —
they don't exist in the audio. But the reconstruction loss (MSE, used by all four model
families — SNN/LSTM/GRU/Transformer-AE) can't tell "real silence" from "padding": every
element of every window, fake or not, counted equally toward the score. The network was
being penalized for failing to "reconstruct" samples that were never part of the
recording.

**A concrete example.** `window_size=8`, one recording is 12 samples long:

```
recording:         [ 3  7  2  9 | 10 20 30 40 ]           <- 12 real samples
window 0 (take=8):  3  7  2  9 10 20 30 40                 full, no padding
window 1 (take=4): 10 20 30 40  0  0  0  0                 4 real + 4 FAKE zeros
                    └──────────┘ └────────┘
                    real prefix   padding (never recorded)
```

`WindowMetadata::valid_length` records this per window: `8` for window 0, `4` for
window 1. `make_activity_mask(valid_length, window_size)` turns that single int into the
tensor a loss/metric needs: `[1,1,1,1,0,0,0,0]` for window 1, all-ones for window 0.
EEG (`eegmmidb`/`siena`) and MIT-BIH loaders never emit this case — their windowing loop
*drops* a trailing partial window instead of padding it, so `valid_length == window_size`
always for every window they produce.

**A second, independent bug found while fixing this.** Before this fix, z-score
normalization ran on the whole *padded* window (real + fake zeros together) — window 1's
mean/std above would have been computed over `[10,20,30,40,0,0,0,0]`, not just the real
`[10,20,30,40]`. Two problems follow: the real samples get normalized against statistics
that don't describe them, and — worse — the padded tail is no longer even literally zero
after normalization (it becomes `(0 - mean) / std` of the *mixed* population), so it can't
even be recognized as padding downstream by inspecting the values. The fix normalizes only
the real prefix, against its own mean/std, and leaves the padded tail at literal `0.0`.

**The mechanism.** `make_activity_mask(valid_length, window_size)` (`Meeting01Encoding.hpp`)
is a pure, content-agnostic function — it doesn't know or care what "window" means, it just
returns a `(window_size, 1)` tensor of 1s then 0s. That's deliberate: `make_reconstruction_target`
and `to_lstm_frames` (the two functions that already reshape the *target* into whatever
shape a given model family needs — time-major replication for the SNN, frame-reshaping for
LSTM/GRU/Transformer) are ALSO pure structural transforms with no notion of "signal" built
in. Passing the raw mask through the *same* calls used for the target produces a mask of
matching shape for free, with zero new reshape logic:

```cpp
const Tensor mask = to_lstm_frames(
    make_activity_mask(meta.valid_length, window_size), frame_size);   // LSTM/GRU/Transformer
const Tensor mask = make_reconstruction_target(
    make_activity_mask(meta.valid_length, window_size), time_steps);   // SNN
```

`MSELossImpl::set_mask(mask)` (`include/layers/losses/MSELoss.hpp`) then restricts both
`forward()` and `backward()` to `sum(mask ⊙ (pred-target)²) / sum(mask)` instead of the
plain element-count mean — a masked-out element contributes to neither the reported loss
nor the gradient, however large its residual. `Trainer::fit_supervised_masked()`
(`src/core/training/Trainer.hpp`) threads a per-sample mask through the training loop
alongside input/target; it is purely additive (a new method, a new `SampleTriple` type,
SFINAE-detected `set_mask` support) so every existing `Trainer`/`MSELossImpl` caller in the
framework — `thesis`, `autoencoderRunner`, core tests — is unaffected unless it opts in.
`mse_between_masked`/`mae_between_masked` (`include/statistics/reconstruction_metrics.hpp`)
apply the identical masked formula to evaluation and per-window-error reporting, so a
model's reported test-set MSE is computed the same way its training loss was — without
this, training would correctly ignore the padding while the *reported* metrics still
silently included it, producing a misleading train/test gap that isn't real.

**Failure mode: silent, and specific to FSDD/AudioMNIST's last window per recording.**
Every other window of every dataset is completely unaffected (`valid_length == window_size`
there, so the mask is all-ones and the masked and unmasked computations coincide exactly).
Only the trailing, padded window of a variable-length FSDD/AudioMNIST recording — a small
fraction of the total — was silently training and scoring against fabricated zero content.

### Denoising-autoencoder corruption

**The problem.** A vanilla autoencoder trained with `target = input` has a trivial escape
hatch: given enough capacity, it can learn something close to the identity function and
still score a good reconstruction loss without ever building a compressed, general
representation of the signal — it's just copying. Vincent et al. (2008, ICML; 2010, JMLR)
close that escape hatch: corrupt the network's INPUT with noise, but keep the loss TARGET
clean. The network can no longer copy its way to a low loss — it has to recover the clean
signal from a corrupted one, which forces the bottleneck to encode signal structure instead
of exact sample values.

**A concrete example.** `denoising_noise_std=0.05`, one post-z-score FSDD sample value
`x=0.42`:

```
clean signal:              0.42
                              │ GaussianNoise(std=0.05), one draw ~ N(0, 0.05²) = -0.03
                              ▼
corrupted encoder input:   0.42 + (-0.03) = 0.39     <- what the network SEES
reconstruction target:     0.42                       <- what the loss SCORES against, unchanged
```

The network is graded on how close its output gets to `0.42`, having only ever seen `0.39`.
It cannot get there by copying its input — it has to have learned enough about the signal's
structure, across the whole training set, to denoise.

**Where corruption is applied, and where it is not.**

```
train_samples[i]  (clean analog window)
        │
        ├─ apply_noise=true ──> GaussianNoise ──> encode_sample ──> ENCODER INPUT (train)
        │
        └───────────────────────────────────────> make_reconstruction_target / to_lstm_frames
                                                    ──> TARGET (train)  <- always clean
                                                    ──> MASK   (train)  <- always clean (see Activity mask, above)

val_samples[i]    (clean analog window)
        └─ apply_noise=false ─> encode_sample ──> ENCODER INPUT (val)   <- also clean
                                                    TARGET / MASK (val) <- clean
```

Validation input is never corrupted (`apply_noise=false` in `make_triples`,
`Meeting01Training.cpp`/`Meeting01AeCommon.hpp`): the comparison across model families is
about how well each one reconstructs the REAL signal, not how well it denoises a synthetic
noise distribution nobody will see at inference. Only the training encoder input is
corrupted; the target and the activity mask are always built from the clean sample.

**The confusable pair: per-epoch resampling vs. per-run fixed noise.**

|                            | Per-epoch resampling (rejected) | Per-run fixed noise (chosen) |
|---|---|---|
| When noise is drawn | fresh draw every epoch, via `Trainer::sample_transform_` | once, when `make_triples` builds the `(input, target, mask)` triples, before the epoch loop starts |
| What it perturbs | the ALREADY-ENCODED tensor — post spike-conversion for SNN, post frame-reshape for LSTM/GRU/Transformer | the RAW analog signal, before `encode_sample` |
| Effect on a spike train | adds Gaussian noise to 0/1 spike values — not physically meaningful, wrong semantic layer | none directly: noise perturbs the analog signal, and the SAME encoding pipeline converts the (now slightly different) signal to spikes, exactly as real sensor noise would |
| Consistency across families | `sample_transform_` exists only on `Trainer`, unreachable before encoding | identical mechanism (same `GaussianNoise` call, same position relative to `encode_sample`) for all 4 families |
| `Trainer` changes needed | none — hook already exists, but at the wrong layer | none |

`Trainer::sample_transform_` (`set_sample_transform()`) already exists and IS applied fresh
every batch/epoch — but only to the tensor AFTER encoding. For SNN that means after
Poisson/latency conversion to spikes, where "add Gaussian noise" no longer means "the
microphone/electrode picked up some noise": it perturbs 0/1 spike values into non-binary
floats, which the spike machinery downstream was never built to consume. Applying noise to
the raw analog signal once per run, before `encode_sample`, keeps the corruption physically
meaningful and identical across all four families being compared — the same reasoning that
already fixed poisson encoding to one draw per run instead of resampling every epoch (see
`.wiki/Experiments/Meeting01.md`).

**Mechanism.** `GaussianNoise` (`include/utility/GaussianNoise.hpp`), a new
`nn::transforms::ITransform`: elementwise `out = x + N(0, std²)`, with a private
`std::mt19937` seeded once at construction and advanced on every call — the same
stateful-callable contract as `RandomCrop`/`RandomIndexCrop` (see Determinism, above).
`denoising_noise_std=0.0f` (the default) makes `operator()` return its input completely
unchanged: `std::normal_distribution` at `stddev=0` is a standard-library precondition
violation (UB), so the class short-circuits instead of constructing that distribution. This
is why every profile written before this feature existed trains byte-identically to before.

**Failure mode: silent, and directional.** If corruption were accidentally applied to the
TARGET as well as the input — or applied after `make_reconstruction_target`/`to_lstm_frames`
instead of before `encode_sample` — the network would be asked to reconstruct a noisy
target. Training would proceed, the loss curve would look completely normal, and the number
the GA optimizes against would still look reasonable — but it would no longer measure
denoising ability at all: a model that perfectly reproduces the corruption would now score
best, the opposite of the property this technique exists to encourage. Nothing about this
fails loudly. It produces a plausible, lower-than-expected training loss (matching noise is
easier than removing it) that would only surface by inspecting reconstructions directly, or
by noticing the val/test reconstruction metrics — computed on the clean target, see above —
diverge suspiciously from the training loss.

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
