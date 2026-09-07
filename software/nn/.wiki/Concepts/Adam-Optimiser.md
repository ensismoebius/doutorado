# Adam Optimiser

Adam (Adaptive Moment Estimation) combines adaptive learning rates with momentum for robust optimization.

## Theoretical Background

Adam maintains two exponential moving averages [1]:

### First Moment (Momentum)

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

Where $g_t$ is the gradient at time $t$.

### Second Moment (Variance)

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

### Bias Correction

Both moments are initialized to zero, causing biased early estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Update Rule

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\eta$ | 0.001 | Learning rate |
| $\beta_1$ | 0.9 | Momentum decay |
| $\beta_2$ | 0.999 | Variance decay |
| $\epsilon$ | $10^{-8}$ | Numerical stability |

## How It Is Implemented Here

```cpp
// File: include/optimizers/Adam.hpp
struct Adam : public Optimizer
{
    float learning_rate;
    float decay_rate_moment1; // beta1, typically 0.9
    float decay_rate_moment2; // beta2, typically 0.999
    float epsilon;
    int delta_t;              // step counter

    std::vector<Tensor> moment1; // first moment (m)
    std::vector<Tensor> moment2; // second moment (v)

    auto step(std::span<Tensor*> paramsList) -> void override
    {
        delta_t += 1;
        for (size_t i = 0; i < paramsList.size(); ++i)
        {
            auto& param = *paramsList[i];
            const float lr_i = learning_rate * (i < lr_scales_.size() ? lr_scales_[i] : 1.0f);
            Tensor grad = param.grad();

            moment1[i] = moment1[i].multiply_scalar(decay_rate_moment1)
                             .add(grad.multiply_scalar(1.0f - decay_rate_moment1));
            moment2[i] = moment2[i].multiply_scalar(decay_rate_moment2)
                             .add(grad.multiply(grad).multiply_scalar(1.0f - decay_rate_moment2));

            float bc1 = 1.0f - std::pow(decay_rate_moment1, delta_t);
            float bc2 = 1.0f - std::pow(decay_rate_moment2, delta_t);
            auto m_hat = moment1[i] / bc1;
            auto v_hat = moment2[i] / bc2;

            param = param.add((m_hat.multiply_scalar(lr_i)
                                   / v_hat.sqrt().add_scalar(epsilon)).multiply_scalar(-1.0f));
        }
    }
    // Real step() also applies decoupled weight decay (AdamW) before the update
    // and takes a fused single-kernel fast path when the backend provides one —
    // trimmed here for clarity.
};
```

## Data Flow

```mermaid
flowchart LR
    subgraph Gradient
        g[Gradient g_t]
    end

    subgraph Moments
        m[Update m<br/>β1⋅m + (1-β1)⋅g]
        v[Update v<br/>β2⋅v + (1-β2)⋅g²]
    end

    subgraph Bias
        mhat[m̂ = m/(1-β1^t)]
        vhat[v̂ = v/(1-β2^t)]
    end

    subgraph Update
        step[θ -= η⋅m̂/(√v̂ + ε)]
    end

    g --> m
    g --> v
    m --> mhat
    v --> vhat
    mhat --> step
    vhat --> step
```

## Usage Example

```cpp
// File: include/optimizers/Adam.hpp
#include "optimizers/Adam.hpp"

// Create optimizer
Adam optimizer(0.001f, 0.9f, 0.999f, 1e-8f);

// Attach to model parameters
optimizer.attach(model.params());

// Training loop
for (int epoch = 0; epoch < epochs; ++epoch)
{
    for (auto& batch : dataloader)
    {
        optimizer.zero_grad();
        auto output = model.forward(batch, true);
        auto loss = criterion(output, target);
        model.backward(loss.grad());
        optimizer.step(model.params());
    }
}
```

## Common Pitfalls

1. **Learning Rate**: Default 0.001 works for most; tune for your problem

2. **Weight Decay**: Use AdamW variant for regularization

3. **Epsilon**: Don't change unless you have numeric issues

4. **Convergence**: Can get stuck in local minima; try SGD fallback

## See Also

- [Optimizers](../Core/Optimizers.md) - Other optimizers
- [Weight-Initialisation](../Concepts/Weight-Initialisation.md) - Combined with initialization

## References

[1] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. on Learning Representations (ICLR)*, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[2] S. J. Reddi, S. Kale, and S. Kumar, "On the convergence of Adam and beyond," in *Proc. 6th Int. Conf. on Learning Representations (ICLR)*, 2018. [Online]. Available: https://arxiv.org/abs/1904.09237