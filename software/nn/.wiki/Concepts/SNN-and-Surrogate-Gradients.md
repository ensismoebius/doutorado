# Spiking Neural Networks and Surrogate Gradients

> **Plain language version:** [SNN and Surrogate Gradients — Plain Language Guide](./Plain/SNN-and-Surrogate-Gradients.md)

Spiking Neural Networks (SNNs) are the third generation of neural networks, using discrete spikes instead of continuous values for information processing.

## Theoretical Background

### Leaky Integrate-and-Fire (LIF) Neuron

The continuous-time LIF equation describes membrane dynamics [7]:

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + I(t), \quad \tau = RC$$

Discretised with time step $\Delta t$, the decay factor $\beta = e^{-\Delta t / \tau}$ gives:

$$V[t] = \beta \cdot V[t-1] + I[t]$$

When $V > V_{th}$, the neuron emits a spike ($s = 1$) and resets its membrane:
- **Hard reset**: $V \leftarrow V_{reset}$ (usually 0)
- **Soft reset**: $V \leftarrow V - V_{th}$ (subtract threshold)

Both reset modes are supported.  Trainable parameters $R$, $C$, and $V_{th}$ are exposed via `params()` so the optimizer can adjust neuron dynamics.

### The Gradient Problem

The spike function is non-differentiable:
$$s = \mathbb{1}(V \geq V_{th}) = \begin{cases} 1 & V \geq V_{th} \\ 0 & \text{otherwise} \end{cases}$$

The true derivative is zero almost everywhere (Dirac delta at threshold), which kills backpropagation.

### Surrogate Gradient Methods

Surrogate gradients replace the non-differentiable spike derivative with a smooth approximation during the backward pass while keeping the exact spike rule in the forward pass [7]:

**Exponential (SuperSpike)**:
$$\frac{\partial s}{\partial V} \approx \frac{1}{\alpha} \exp\!\left(-\frac{|V - V_{th}|}{\alpha}\right)$$

**Boxcar**:
$$\frac{\partial s}{\partial V} \approx \begin{cases} 1 & |V - V_{th}| < w/2 \\ 0 & \text{otherwise} \end{cases}$$

Where $\alpha$ (sharpness) or $w$ (window width) are hyperparameters.

### Spike-Frequency Adaptation

To prevent bursting and improve temporal coding, an adaptation variable $a[t]$ raises the effective threshold after each spike [35]:

$$V_{th,\text{eff}}[t] = V_{th} + a[t]$$
$$a[t] = d \cdot a[t-1] + c \cdot s[t-1]$$

where $d \in (0,1)$ is the decay factor (`adapt_decay`) and $c \geq 0$ is the coupling constant (`adapt_coupling`).  When no spike occurs $a$ decays toward zero; each spike pushes it up by $c$, increasing resistance to subsequent spikes.

### Threshold-Dependent Batch Normalization (tdBN)

Deep SNNs without normalization suffer from membrane-potential explosion across layers.  tdBN [33] normalises the pre-spike potential at each time step and rescales by $V_{th}/\sqrt{T}$ so input magnitude is independent of network depth:

$$\hat{x}_{t,b,f} = \gamma_f \cdot \frac{x_{t,b,f} - \mu_{t,f}}{\sqrt{\sigma^2_{t,f} + \varepsilon}} + \beta_f$$
$$y_{t,b,f} = \hat{x}_{t,b,f} \cdot \frac{V_{th}}{\sqrt{T}}$$

where $\mu$ and $\sigma^2$ are computed over the batch dimension for each $(t, f)$ slice.  $\gamma$ and $\beta$ are learned per-feature affine parameters.

### Poisson Latent Space (SNN-VAE)

For variational spiking autoencoders, the Gaussian VAE reparameterisation is replaced by a Poisson spike process [29, 30]:

1. Encoder outputs logits $z$ (B × F)
2. Rate: $\lambda = \text{softplus}(z)$, ensuring $\lambda > 0$
3. Training sample: $s \sim \text{Poisson}(\lambda \cdot T)$, output $= s/T$
4. KL divergence: $\text{KL}(\text{Poisson}(\lambda) \| \text{Poisson}(\lambda_0)) = \lambda_0 - \lambda + \lambda \log(\lambda/\lambda_0) \geq 0$

   *(Previous wiki entry had the sign reversed — matched a code bug fixed 2026-05-01; see `PoissonLatentTest.KLNonNegative`.)*

Straight-through gradient: $\partial L / \partial z \approx (\partial L / \partial \text{output}) \cdot (1/T) \cdot \sigma(z)$

**Tests:** `TdBNTest.*` (formula, γ/β grad, V_th/√T scaling), `PoissonLatentTest.*` (rate positivity, KL≥0, straight-through grad) — in `src/core/layers/tests/fundamental_mechanisms_gtest.cpp`.

---

## How It Is Implemented Here

### Surrogate Gradient Interface

```cpp
// File: include/layers/spiking/ISurrogateGradient.hpp
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
// File: include/layers/spiking/ExponentialSurrogate.hpp
class ExponentialSurrogate : public ISurrogateGradient
{
    float sharpness_ = 1.0f;
public:
    explicit ExponentialSurrogate(float sharpness = 1.0f) : sharpness_(sharpness) {}
    float calculate_scalar(float v_mem_pre_spike, float voltage_threshold) const override
    {
        float diff_abs = std::abs(v_mem_pre_spike - voltage_threshold);
        return (1.0f / sharpness_) * std::exp(-diff_abs / sharpness_);
    }
};
```

### Single-Step Spiking Neuron (LifImpl)

`LifImpl` keeps persistent membrane state across sequential `forward()` calls.
It is used in single-step pipelines where the caller advances the simulation manually.

```cpp
// File: include/layers/spiking/Lif.hpp
template <typename Backend>
struct LifImpl : public Module<Backend>
{
    float time_step = 1.0F;
    Tensor resistance;       // trainable: 1×1
    Tensor capacitance;      // trainable: 1×1
    Tensor voltage_threshold; // trainable: 1×1
    Tensor v_mem;            // persistent membrane state (B×F)

    // Spike-frequency adaptation
    float adapt_decay    = 0.9F;  // decay factor d ∈ (0,1)
    float adapt_coupling = 0.0F;  // coupling c ≥ 0 (0 = disabled)
    Tensor adapt_a;               // adaptation variable (B×F)

    // Construction: all parameters have defaults; adaptation is off by default.
    explicit LifImpl(float time_step_ = 1.0F,
        float resistance_ = 1.0F, float capacitance_ = 1.0F,
        float voltage_threshold_ = 1.0F,
        bool reset_zero_ = true, float reset_potential_ = 0.0F,
        std::shared_ptr<ISurrogateGradient> surrogate_grad = make_shared<ExponentialSurrogate>(),
        float adapt_decay_ = 0.9F, float adapt_coupling_ = 0.0F);

    // Forward: V[t] = β·V[t-1] + I[t]; spike if V > V_th + adapt_a; reset V
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    // Backward: uses surrogate gradient for dS/dV
    auto backward(const Tensor& grad_output) -> Tensor override;
};
```

Key forward mechanics:
- $\beta = \exp(-\Delta t / (RC))$, clamped to avoid NaN when $R$ or $C \leq 0$
- Effective threshold: $V_{th,\text{eff}} = V_{th} + a$ (when `adapt_coupling > 0`)
- `adapt_a` decays by `adapt_decay` each step; increments by `adapt_coupling` on spike
- `reset_state()` clears both `v_mem` and `adapt_a`

### BPTT Spiking Neuron (LifBPTTImpl)

`LifBPTTImpl` unrolls the full time sequence in a single `forward()` call and
computes exact BPTT gradients through the recurrence.  Input shape: `(T*B, F)`.

```cpp
// File: include/layers/spiking/LifBPTT.hpp
template <typename Backend>
struct LifBPTTImpl : public Module<Backend>
{
    int time_steps;          // T
    float time_step = 1.0F;  // Δt
    Tensor resistance, capacitance, voltage_threshold;  // trainable 1×1
    Tensor v_mem;            // persistent state across forward() calls (B×F)
    Tensor v_mem_history;    // pre-spike V cache for BPTT (T*B × F)
    Tensor v_post_history;   // post-reset V cache: eliminates reconstruction drift
    bool readout_mode = false; // if true, emit V directly (no spike/reset)

    // Spike-frequency adaptation
    float adapt_decay    = 0.9F;
    float adapt_coupling = 0.0F;
    Tensor adapt_a_bptt_;  // persistent across forward() calls (B×F)

    explicit LifBPTTImpl(int time_steps_,
        float time_step_ = 1.0F,
        float resistance_ = 1.0F, float capacitance_ = 1.0F,
        float voltage_threshold_ = 1.0F,
        bool reset_zero_ = true, float reset_potential_ = 0.0F,
        bool readout_mode_ = false,
        std::shared_ptr<ISurrogateGradient> surrogate_grad = make_shared<ExponentialSurrogate>(),
        float adapt_decay_ = 0.9F, float adapt_coupling_ = 0.0F);
};
```

BPTT backward loop (reverse time):
- `grad_v_pre = grad_out * surr + grad_from_next * β * dvpost_dvpre`
- `dL/dV_th` accumulates both direct spike term and recurrent reset-path term
- `dL/dR`, `dL/dC` use `v_post_history` to avoid reconstruction drift

### Threshold-Dependent Batch Normalization (ThresholdDependentBatchNormImpl)

```cpp
// File: include/layers/spiking/ThresholdDependentBatchNorm.hpp
template <typename Backend>
class ThresholdDependentBatchNormImpl : public Module<Backend>
{
public:
    float voltage_threshold = 1.0F;  // V_th of downstream LIF layer
    int time_steps = 1;              // T
    float eps = 1e-5F;
    Tensor gamma;  // learned per-feature scale (1×F)
    Tensor beta;   // learned per-feature shift (1×F)

    explicit ThresholdDependentBatchNormImpl(
        size_t num_features, float vth = 1.0F, int T = 1, float eps_ = 1e-5F);

    // forward: per-step per-feature BN, scaled by V_th/sqrt(T)
    // backward: standard BN gradient through affine transform
};
```

Usage — insert between `Linear` and `LifBPTT` in deep SNN encoders:
```cpp
Linear<Backend>    fc(128, 64);
ThresholdDependentBatchNormImpl<Backend> tdbn(64, /*vth=*/1.0f, /*T=*/10);
LifBPTTImpl<Backend> lif(/*time_steps=*/10, ...);

auto h = fc.forward(input, true);
h = tdbn.forward(h, true);          // normalise + threshold-dependent scale
auto spikes = lif.forward(h, true); // LIF receives well-conditioned input
```

### Poisson Latent Layer (PoissonLatentLayerImpl)

```cpp
// File: include/layers/spiking/PoissonLatentLayer.hpp
template <typename Backend>
class PoissonLatentLayerImpl : public Module<Backend>
{
public:
    int time_steps = 1;
    float prior_rate = 0.1F;   // λ₀ for KL divergence
    float beta_kl = 1.0F;      // β-VAE weighting

    // forward(train):  λ=softplus(z); s~Poisson(λ*T); return s/T
    // forward(infer):  return λ (no sampling)
    float kl_loss() const;          // KL term, add β*kl_loss() to total loss
    const Tensor& last_rates() const; // λ values for sparsity logging

    explicit PoissonLatentLayerImpl(int T = 1, float prior_rate = 0.1F, float beta_kl = 1.0F);
};
```

Training pattern for β-SVAE:
```cpp
PoissonLatentLayerImpl<Backend> latent(/*T=*/10);
auto s = latent.forward(encoder_out, true);    // reparameterized spike sample
auto recon = decoder.forward(s, true);
float total_loss = mse_loss(recon, target) + beta * latent.kl_loss();
```

---

## Data Flow

```mermaid
flowchart TB
    subgraph Input
        I[Input Current I(t)]
    end

    subgraph Integrate
        beta["V[t] = β·V[t-1] + I[t]"]
    end

    subgraph Adapt
        a["V_th_eff = V_th + adapt_a"]
    end

    subgraph Fire
        check{V > V_th_eff?}
        spike[Spike s=1]
        no_spike[No spike s=0]
    end

    subgraph Reset
        reset["V = V_reset (hard) or V-V_th (soft)"]
        incr["adapt_a += adapt_coupling"]
    end

    subgraph Gradient
        sg[Surrogate ∂s/∂V]
    end

    I --> beta
    beta --> a --> check
    check -->|yes| spike --> reset --> incr
    check -->|no| no_spike
    beta -.->|backward| sg
```

---

## Common Pitfalls

1. **Argument order in constructors**: `surrogate_grad` comes before `adapt_decay` / `adapt_coupling`.  Passing a `shared_ptr` where a `float` is expected is a compile error.

2. **Threshold too low/high**: Too low → excessive firing and gradient saturation; too high → dead neurons (zero gradient).

3. **Adaptation without decay**: Setting `adapt_coupling > 0` with `adapt_decay = 1.0` causes the threshold to grow without bound.

4. **BPTT vs single-step**: Use `LifBPTTImpl` when you need exact parameter gradients for R/C/V_th.  Use `LifImpl` for shallow/feed-forward pipelines where BPTT is approximated externally.

5. **Loss–encoding mismatch**: Rate-coded outputs require `SpikeCountLoss`; latency-coded outputs require `SpikeTimeLoss`.  Mixing them reverses gradient sign for some neurons.

---

## See Also

- [LSTM and BPTT](./LSTM-and-BPTT.md) — Related recurrent implementations
- [Layers](../Core/Layers.md) — Full layer catalogue
- [Weight Initialisation](./Weight-Initialisation.md) — Initialization for SNN
- [Spike Rate Regularization](./Spike-Rate-Regularization.md) — Preventing dead/bursting neurons
- [Spike Encoding](./Spike-Encoding.md) — Rate vs latency coding and matching loss functions

---

## References

[7] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate gradient learning in spiking neural networks," *IEEE Signal Process. Mag.*, vol. 36, no. 6, pp. 51–63, Nov. 2019. [Online]. Available: https://arxiv.org/abs/1901.09948

[26] W. Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence," *Science Advances*, vol. 9, no. 40, eadi1480, 2023. [Online]. Available: https://www.science.org/doi/10.1126/sciadv.adi1480

[27] W. Gerstner and W. M. Kistler, *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press, 2002.

[29] K. Kamata et al., "Fully spiking variational autoencoder," in *Proc. AAAI Conf. Artificial Intelligence*, 2022.

[30] C. Chen et al., "ESVAE: An efficient spiking variational autoencoder with reparameterizable Poisson spiking sampling," arXiv:2310.14839, 2024. [Online]. Available: https://arxiv.org/html/2310.14839v2

[33] Y. Zheng et al., "Going deeper with directly-trained larger spiking neural networks," in *Proc. AAAI Conf. Artificial Intelligence*, 2021. [Online]. Available: https://arxiv.org/abs/2011.05280

[35] J.-C. Zhao et al., "MPD-ATP: Multi-phase dynamics adaptive threshold plasticity for spiking neural networks," *IEEE Trans. Neural Netw. Learn. Syst.*, 2025. [Online]. Available: https://ieeexplore.ieee.org/document/11264550/

[36] T. Limbacher et al., "AdaLi: Adaptive surrogate gradient for spiking neural networks," *Frontiers in Neuroscience*, 2026. [Online]. Available: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1795946/full
