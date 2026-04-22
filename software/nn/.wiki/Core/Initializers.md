# Initializers

Weight initialization strategies to prevent vanishing/exploding gradients.

## Theoretical Background

Proper initialization ensures activations and gradients maintain reasonable magnitudes across layers. Without it, deep networks fail to train effectively [3].

### Xavier/Glorot Initialization

Designed for sigmoid/tanh activations. Weights sampled from:

$$W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right)$$

Or equivalently from Gaussian $\mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$.

This maintains variance of activations: $\text{Var}(y) = \text{Var}(x)$ across layers [3].

### Kaiming/He Initialization

Designed for ReLU and variants. Weights sampled from:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

The factor of 2accounts for ReLU zeroing half the values [4].

For Leaky ReLU with leak rate $\alpha$:

$$W \sim \mathcal{N}\left(0, \frac{1}{1 - \alpha^2}\right)$$

## How It Is Implemented Here

```cpp
// File: include/nn/initializers/xavier.hpp
class XavierInitializer
{
public:
    static void initialize(Tensor& weights, unsigned int seed = std::random_device{}())
    {
        auto [fan_in, fan_out] = get_fan(weights);
        float bound = std::sqrt(6.0f / (fan_in + fan_out));

        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-bound, bound);

        for (size_t i = 0; i < weights.rows(); ++i)
        {
            for (size_t j = 0; j < weights.cols(); ++j)
            {
                weights.at(i, j) = dist(gen);
            }
        }
    }
};
```

```cpp
// File: include/nn/initializers/kaiming_snn.hpp
class KaimingSNNInitializer
{
public:
    static void initialize(Tensor& weights, float leak_rate = 0.0f,
                     unsigned int seed = std::random_device{}())
    {
        auto [fan_in, fan_out] = get_fan(weights);

        float std = std::sqrt(2.0f / fan_in);
        if (leak_rate > 0.0f)
        {
            std /= std::sqrt(1.0f - leak_rate * leak_rate);
        }

        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, std);

        for (size_t i = 0; i < weights.rows(); ++i)
        {
            for (size_t j = 0; j < weights.cols(); ++j)
            {
                weights.at(i, j) = dist(gen);
            }
        }
    }
};
```

## Data Flow

```mermaid
flowchart TB
    subgraph Init
        dim[Get fan_in<br/>fan_out]
    end

    subgraph Compute
        calc[Calculate<br/>std/bound]
    end

    subgraph Sample
        rand[Sample<br/>Normal/Uniform]
    end

    subgraph Assign
        fill[Fill<br/>Weight Matrix]
    end

    dim --> calc
    calc --> rand
    rand --> fill
```

## Usage Example

```cpp
// File: src/core/layers/dense/Linear.hpp
#include "nn/initializers/xavier.hpp"

template <typename Backend>
class Linear : public Module<Backend>
{
    Tensor weights_;

public:
    Linear(size_t in_features, size_t out_features)
    {
        weights_ = Tensor(in_features, out_features);
        XavierInitializer::initialize(weights_);
        bias_ = Tensor(1, out_features);
        bias_.fill(0.0f);
    }
};
```

## Common Pitfalls

1. **Wrong Initialization for Activation**: Use Xavier for sigmoid/tanh, Kaiming for ReLU

2. **Bias Initialization**: Set biases to zero initially

3. **Inconsistent Shape**: Ensure `fan_in`/`fan_out` computed correctly for conv layers

4. **Random Seed**: Always set seed for reproducibility in experiments

## See Also

- [Layers](./Layers.md) - Using initialized weights
- [Weight-Initialisation](../Concepts/Weight-Initialisation.md) - Detailed theory
- [Optimizers](./Optimizers.md) - Optimizing after initialization

## References

[1] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in Proc. 13th Int. Conf. Artificial Intelligence and Statistics (AISTATS), 2010, pp. 249–256.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2015. [Online]. Available: https://arxiv.org/abs/1502.01852