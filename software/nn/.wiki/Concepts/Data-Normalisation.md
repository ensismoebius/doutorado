# Data Normalisation

Normalizing input data is crucial for neural network training.

## Theoretical Background

### Z-Score Normalization

Standardizes to zero mean, unit variance:

$$x' = \frac{x - \mu}{\sigma}$$

Where $\mu$ is mean and $\sigma$ is standard deviation.

### Min-Max Scaling

Scales to [0, 1] range:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Per-Feature vs Per-Sample

- **Per-feature**: Normalize each column independently (standard for MLPs)
- **Per-sample**: Normalize each sample (row) independently (common in signal processing)

## Implementation

### Z-Score Normalizer

```cpp
// File: include/nn/utility/AudioMeanStdNormalize.hpp
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

### EEG Window Z-Score

```cpp
// File: include/nn/utility/EEGWindowZScore.hpp
// Per-window (row) z-score normalization
// Stateless: computes statistics on-the-fly
```

### Fused Modality Transform

```cpp
// File: include/nn/utility/FusedModalityTransform.hpp
// Different normalization for EEG and audio in fused tensor
```

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

## References

[1] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, 2015, pp. 448–456. [Online]. Available: https://arxiv.org/abs/1502.03167

[2] F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A review of classification algorithms for EEG-based brain-computer interfaces: A 10-year update," *J. Neural Eng.*, vol. 15, no. 3, p. 031005, 2018. [Online]. Available: https://doi.org/10.1088/1741-2552/aab2f2