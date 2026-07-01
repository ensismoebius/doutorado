# Data Normalisation

Normalizing input data is crucial for neural network training.

## Theoretical Background

### Why Normalize At All

Unnormalized inputs with very different scales make gradient descent unstable: the gradient with respect to a large-magnitude feature dominates the update direction, so the optimizer effectively ignores small-magnitude features until the large ones are fit. Normalizing inputs to comparable scale (and, ideally, zero mean) removes this asymmetry and is one of the oldest documented tricks for making backpropagation converge reliably [55]. This is a *different* mechanism from Batch Normalization [56], which renormalizes **hidden-layer** activations *during* training rather than the raw input once before it — the two are complementary, not interchangeable; this project applies the input-side normalization described below and, separately, [Threshold-Dependent Batch Normalization](Threshold-Dependent-Batch-Normalization.md) inside the spiking network itself.

### Z-Score Normalization

Let $x$ be the raw value, $x'$ the normalized value, $\mu$ the mean, and $\sigma$ the standard deviation (both estimated over whatever population — feature or window — is being normalized). Z-score standardizes to zero mean, unit variance:

$$x' = \frac{x - \mu}{\sigma}$$

### Min-Max Scaling

Let $x$ be the raw value, $x'$ the normalized value, and $x_{min}$, $x_{max}$ the minimum and maximum observed values (over the same population). Min-max scales to the $[0,1]$ range:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Per-Feature vs Per-Sample — and Why This Project Uses Both

- **Per-feature** (column-wise): $\mu,\sigma$ estimated once per feature over the training set, then applied unchanged to every sample. Standard for tabular/MLP inputs whose columns keep a fixed meaning across samples.
- **Per-sample** (row-wise): $\mu,\sigma$ recomputed independently for every sample. Common in signal processing when each recording can have its own baseline/scale.

This project does **not** pick one strategy globally — it picks per modality, and the choice is grounded in how each signal actually behaves:

- **Audio features → per-feature, fit on train.** LFCC/Mel/Bark band energies (see [LFCC](LFCC.md)) keep a fixed physical meaning per column across the whole corpus — column $j$ is always the same frequency band or cepstral order. A single $(\mu_j,\sigma_j)$ estimated from training data therefore generalizes: this is exactly **Cepstral Mean and Variance Normalization (CMVN)**, the classical technique for noise-robust speech recognition [57].
- **EEG windows → per-sample (per-window), no fitting.** EEG amplitude is not stable across recordings: baseline drift, electrode impedance, and amplifier gain vary session to session and subject to subject, so a single global $(\mu,\sigma)$ fit on training data would not correct a new session's offset [58]. Normalizing each window against its own mean/std removes that per-window DC offset and scale on the spot — at the cost of discarding absolute amplitude, which is largely unreliable in raw EEG anyway [58].

### Fitting Only on Training Data

Whenever statistics ($\mu,\sigma$, or min/max) are estimated from data (the per-feature/audio case above), they must be estimated **only** from the training split and then applied, unchanged, to validation/test data. Estimating them from the full dataset (including test) is a form of **data leakage** — information about the test set influences preprocessing, which inflates validation metrics in a way that will not hold up on genuinely unseen data [59]. This is why `AudioMeanStdNormalize` below is a *stateful, fit/transform* object rather than a stateless function: `accumulate()`/`finalize()` must only ever see training batches. Per-window EEG normalization sidesteps this risk entirely — it fits nothing, so there is no leakage surface to worry about.

## Implementation

### Z-Score Normalizer

```cpp
// File: include/utility/AudioMeanStdNormalize.hpp
class AudioMeanStdNormalize : public ITransform
{
    Tensor mean_;      // Column means
    Tensor std_;      // Column standard deviations
    bool fitted_ = false;

    void accumulate(const Tensor& batch);  // Accumulate statistics
    void finalize();                   // Compute mean/std
    auto operator()(const Tensor& x) const -> Tensor;

    auto is_fitted() const -> bool;
};
```

`accumulate()` must only be called on training batches, then `finalize()` computes the column mean/std once; `operator()` (the transform step) then applies those fixed statistics to any batch — training, validation, or test — without ever recomputing them. This is the fit/transform split that keeps the CMVN-style normalization leakage-free.

### EEG Window Z-Score

```cpp
// File: include/utility/EEGWindowZScore.hpp
// Per-window (row) z-score normalization
// Stateless: computes statistics on-the-fly
```

No `accumulate()`/`finalize()` here by design — each call to `operator()` computes that window's own mean/std over its columns and normalizes against them immediately. Nothing is fit, so there is no train/test distinction to get wrong.

### Fused Modality Transform

```cpp
// File: include/utility/FusedModalityTransform.hpp
// Different normalization for EEG and audio in fused tensor
```

Applies the two strategies above to the corresponding column blocks of a single fused (EEG + audio) tensor — each modality is normalized by the strategy grounded above for that signal type, not by a one-size-fits-all rule.

## Data Flow

```mermaid
flowchart LR
    subgraph Fit
        train[Training Data]
        acc[accumulate()]
        mean[Mean]
        std[Std]
    end

    subgraph Transform
        test[Test Data]
        norm[Normalize]
        out[Normalized]
    end

    train --> acc
    acc --> mean --> norm
    acc --> std --> norm
    test --> norm --> out
```

## See Also

- [DataLoaders](../Core/DataLoaders.md) - Integration with data pipeline
- [Layers](./Layers.md) - Normalization affects training
- [Threshold-Dependent Batch Normalization](Threshold-Dependent-Batch-Normalization.md) — the complementary, *internal* normalization used inside the SNN itself
- [LFCC](LFCC.md) — the audio features normalized per-feature

## References

[55] Y. LeCun, L. Bottou, G. B. Orr, and K.-R. Müller, "Efficient BackProp," in *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science, vol. 1524, Springer, 1998, pp. 9–50.

[56] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, 2015, pp. 448–456. [Online]. Available: https://arxiv.org/abs/1502.03167

[57] O. Viikki and K. Laurila, "Cepstral domain segmental feature vector normalization for noise robust speech recognition," *Speech Communication*, vol. 25, no. 1–3, pp. 133–147, 1998.

[58] F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A review of classification algorithms for EEG-based brain-computer interfaces: A 10-year update," *J. Neural Eng.*, vol. 15, no. 3, p. 031005, 2018. [Online]. Available: https://doi.org/10.1088/1741-2552/aab2f2

[59] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, pp. 1–21, 2012.
