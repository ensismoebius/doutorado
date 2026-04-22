# Spiking Neural Networks and Surrogate Gradients

Spiking Neural Networks (SNNs) are the third generation of neural networks, using discrete spikes instead of continuous values for information processing.

## Theoretical Background

### Leaky Integrate-and-Fire (LIF) Neuron

The LIF neuron integrates input current and fires when membrane potential exceeds a threshold [7]:

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + I(t)$$

Where:
- $V$ is membrane potential
- $\tau$ is time constant
- $I$ is input current

When $V > V_{th}$, the neuron emits a spike ($s = 1$) and resets $V$.

### The Gradient Problem

The spike function is non-differentiable:
$$s = \mathbb{1}(V - V_{th}) = \begin{cases} 1 & \text{if } V \geq V_{th} \\ 0 & \text{otherwise} \end{cases}$$

The derivative is zero everywhere except at the threshold (Dirac delta), making gradient-based learning impossible.

### Surrogate Gradient Methods

Surrogate gradients replace the problematic derivative with a smooth approximation [7]:

**Exponential (SuperSpike)**:
$$\frac{\partial s}{\partial V} \approx \frac{1}{\alpha} \exp\left(-\frac{|V - V_{th}|}{\alpha}\right)$$

**Boxcar**:
$$\frac{\partial s}{\partial V} \approx \begin{cases} 1 & \text{if } |V - V_{th}| < \frac{w}{2} \\ 0 & \text{otherwise} \end{cases}$$

Where $\alpha$ (sharpness) or $w$ (window width) are hyperparameters.

## How It Is Implemented Here

### Surrogate Gradient Interface

```cpp
// File: include/nn/layers/spiking/ISurrogateGradient.hpp
class ISurrogateGradient
{
public:
    virtual auto calculate(
        const Tensor& v_mem_pre_spike, float voltage_threshold) const -> Tensor = 0;

    virtual auto calculate_scalar(
        float v_mem_pre_spike, float voltage_threshold) const -> float = 0;
};
```

### Exponential Surrogate

```cpp
// File: include/nn/layers/spiking/ExponentialSurrogate.hpp
class ExponentialSurrogate : public ISurrogateGradient
{
    float sharpness_ = 1.0f;

public:
    explicit ExponentialSurrogate(float sharpness = 1.0f) : sharpness_(sharpness) {}

    auto calculate_scalar(float v_mem_pre_spike, float voltage_threshold) const -> float override
    {
        float diff_abs = std::abs(v_mem_pre_spike - voltage_threshold);
        return (1.0f / sharpness_) * std::exp(-diff_abs / sharpness_);
    }
};
```

### Spiking Neuron (Leaky)

```cpp
// File: include/nn/layers/spiking/Leaky.hpp
class Leaky : public Module<EigenTensorBackend>
{
    float threshold_ = 1.0f;
    float leak_rate_ = 0.99f;

    auto forward(const Tensor& input, bool requires_grad) -> Tensor override
    {
        // Membrane integration
        membrane_potential_ = membrane_potential_ * leak_rate_ + input;

        // Spike generation using surrogate gradient
        Tensor spikes(membrane_potential_.rows(), membrane_potential_.cols());
        for (size_t i = 0; i < membrane_potential_.rows(); ++i)
        {
            for (size_t j = 0; j < membrane_potential_.cols(); ++j)
            {
                float v = membrane_potential_.at(i, j);
                spikes.at(i, j) = (v >= threshold_) ? 1.0f : 0.0f;
            }
        }

        // Reset after spike
        membrane_potential_ = membrane_potential_ * (1.0f - spikes);

        return spikes;
    }
};
```

## Data Flow

```mermaid
flowchart TB
    subgraph Input
        I[Input Current<br/>I(t)]
    end

    subgraph Integrate
        integrate[Integrate<br/>V += I - leak]
    end

    subgraph Fire
        check{V > V_th?}
        spike[Spike s=1]
        no_spike[No spike s=0]
    end

    subgraph Reset
        reset[V = V_rest]
    end

    subgraph Gradient
        sg[Surrogate<br/>Gradient ∂s/∂V]
    end

    I --> integrate
    integrate --> check
    check -->|yes| spike
    check -->|no| no_spike
    spike --> reset
    reset --> integrate
    integrate -.-> sg
```

## Usage Example

```cpp
// File: src/experiments/03/lib/src/LSTMAutoencoder.cpp
#include "nn/layers/spiking/Leaky.hpp"
#include "nn/layers/spiking/ExponentialSurrogate.hpp"

// Create spiking layer
nn::layers::Leaky leaky_neuron(
    threshold = 1.0f,
    leak_rate = 0.99f,
    std::make_shared<ExponentialSurrogate>(1.0f)
);

// Forward pass produces spikes
Tensor spikes = leaky_neuron.forward(input, true);

// Training with surrogate gradient
// ... backward pass uses surrogate gradient approximation
```

## Common Pitfalls

1. **Threshold Selection**: Too low causes excessive firing; too high prevents learning

2. **Sharpness Parameter**: Too small = narrow gradient (poor learning); too large = broad gradient (unstable)

3. **Temporal Dynamics**: SNNs require careful handling of time steps; ensure proper reset

4. **Surrogate vs True Gradient**: Remember surrogate is approximation; may not match true gradient exactly

## See Also

- [LSTM and BPTT](./LSTM-and-BPTT.md) - Related recurrent implementations
- [Layers](../Core/Layers.md) - Other layer types
- [Weight Initialisation](./Weight-Initialisation.md) - Initialization for SNN

## References

[1] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate gradient learning in spiking neural networks," IEEE Signal Process. Mag., vol. 36, no. 6, pp. 51–63, Nov. 2019. [Online]. Available: https://arxiv.org/abs/1901.09948

[2] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
