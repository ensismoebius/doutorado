# Threshold-Dependent Batch Normalization (tdBN)

Threshold-Dependent Batch Normalization (tdBN) is a normalization layer for deep Spiking Neural Networks (SNNs) introduced by Zheng et al. [33]. It adapts Batch Normalization to the spiking setting by (i) pooling statistics over **both the batch and the time dimension** and (ii) rescaling the normalized pre-activation by the neuron's firing threshold $V_{th}$ instead of to unit variance. In this project it is implemented in [`include/layers/spiking/ThresholdDependentBatchNorm.hpp`](../../include/layers/spiking/ThresholdDependentBatchNorm.hpp) and wired into the Experiment 05 deep spiking classifier.

---

## Motivation

Plain Batch Normalization (BN) was designed for analog (real-valued) activations and normalizes a pre-activation to zero mean and unit variance, $N(0,1)$ [56]. Applying it unchanged inside an SNN is problematic for three reasons [33]:

1. **It ignores the time axis.** An SNN processes a sequence of $T$ time steps; the membrane current of a neuron is a *distribution over both samples and time*. Vanilla BN normalizes each time step (or the batch only) and therefore does not control the statistics that actually drive firing across the whole sequence [33], [49].
2. **It ignores the threshold.** Whether a neuron spikes depends on how its membrane potential compares to $V_{th}$. Normalizing to unit variance is blind to $V_{th}$: if $V_{th}\neq 1$, a unit-variance input either rarely crosses the threshold (too few spikes) or saturates it (too many) [33].
3. **It does not, by itself, preserve gradient norm across depth.** Stacking many spiking layers makes the membrane potential — and the back-propagated gradient — shrink or blow up, which is exactly what prevents training very deep SNNs [33].

tdBN keeps BN's stabilizing effect while fixing all three: it pools over batch *and* time, and it ties the output spread to $V_{th}$ so that a controlled fraction of neurons sits near threshold at every depth [33].

---

## Problem

### Distribution of membrane potentials

For a feature/channel $k$, the pre-spike membrane current $X_k$ across a mini-batch and all time steps forms a distribution. If that distribution drifts far below $V_{th}$, the neuron is silent; if it sits far above, the neuron fires every step. Useful computation happens only when a sensible fraction of the mass is *near* $V_{th}$ [33].

### Effect of the threshold

The LIF firing rule is $s = \mathbb{1}[V > V_{th}]$ (see [Membrane Dynamics](Membrane-Dynamics.md)). A normalization that targets $N(0,1)$ is only "balanced" when $V_{th}=1$. For any other threshold the firing statistics are mismatched, so the normalization target must scale with $V_{th}$ [33].

### Training difficulty: exploding / vanishing gradients

SNNs are trained by Backpropagation Through Time with surrogate gradients (see [SNN and Surrogate Gradients](SNN-and-Surrogate-Gradients.md)). The surrogate derivative $\sigma'(V-V_{th})$ is non-zero only near threshold. Across many layers the product of these factors and the recurrent decay $\beta$ either vanishes (deep network goes silent) or explodes. Zheng et al. analyze this with *Block Dynamical Isometry* and show tdBN keeps each block's gradient norm close to 1, enabling networks of 50+ layers to train directly [33].

### The No-Spike Problem

If a layer's pre-activations fall entirely below $V_{th}$, it emits **no spikes**. Then the surrogate gradient there is ≈ 0, so no learning signal flows back and the layer is permanently dead — the *No-Spike Problem*. tdBN guarantees, by construction, that the pre-activation has standard deviation $\alpha V_{th}$, so a predictable fraction of neurons crosses threshold and the layer keeps firing [33]. This complements the firing-rate regularizer used elsewhere in this project (see [Spike-Rate Regularization](Spike-Rate-Regularization.md)).

> **Not the same failure mode as [Spike-Encoding's "No-Spike Problem"](Spike-Encoding.md#the-no-spike-problem).** Here the cause is the surrogate gradient of the LIF neuron itself going to ≈ 0 far from threshold. The `SpikeTimeLoss` version is caused by how the loss's straight-through estimator indexes into the spike tensor when a predicted unit never fires — a different mechanism, same "silence → zero gradient → stuck" symptom.

---

## Mathematical formulation

Let the time-major pre-activation tensor be $X \in \mathbb{R}^{(T\cdot B)\times F}$, where:

- $T$ — number of simulation time steps (dimensionless),
- $B$ — batch size (number of samples; dimensionless),
- $F$ — number of features/channels,
- $X_{k}$ — the column (channel) $k$, i.e. all $N = T\cdot B$ values of feature $k$ pooled over batch **and** time. Physically these are membrane currents (in the model's normalized voltage units).

**Per-channel statistics** (pooled over the $N = T\cdot B$ rows):

$$\mu_k = \frac{1}{N}\sum_{i=1}^{N} X_{k,i}, \qquad \sigma_k^2 = \frac{1}{N}\sum_{i=1}^{N}\left(X_{k,i}-\mu_k\right)^2$$

- $\mu_k$ — mean of channel $k$ (voltage units),
- $\sigma_k^2$ — (biased) variance of channel $k$ (voltage² units).

**Normalization:**

$$\hat{X}_{k,i} = \frac{X_{k,i}-\mu_k}{\sqrt{\sigma_k^2 + \varepsilon}}$$

- $\hat{X}_{k,i}$ — standardized value (dimensionless, zero mean / unit variance),
- $\varepsilon$ — small constant for numerical stability (voltage², default $10^{-5}$).

**Threshold-dependent affine output** (the tdBN equation of [33]):

$$Y_{k,i} = \gamma_k\,\big(\alpha\,V_{th}\,\hat{X}_{k,i}\big) + \beta_k$$

- $V_{th}$ — firing threshold of the downstream LIF neuron (voltage units),
- $\alpha$ — hyperparameter setting the target spread (dimensionless, paper default $\alpha=1$),
- $\gamma_k,\ \beta_k$ — learnable scale and shift of channel $k$ ($\gamma$ dimensionless, $\beta$ voltage units),
- $Y_{k,i}$ — the normalized current fed to the LIF layer (voltage units).

With the default $\gamma_k=1,\ \beta_k=0$, the output is distributed as $N\!\big(0,(\alpha V_{th})^2\big)$ — the defining property of tdBN [33]. Note that $\alpha V_{th}$ multiplies **only** the normalized term; the shift $\beta_k$ is *not* scaled.

---

## Derivation

tdBN follows from two requirements layered onto standard BN.

**Step 1 — Standard BN.** BN computes $\hat{X} = (X-\mu)/\sqrt{\sigma^2+\varepsilon}$ and $Y=\gamma\hat{X}+\beta$, with $\mu,\sigma^2$ over the batch [56]. This yields $\mathrm{Var}(Y)=\gamma^2$, i.e. $N(0,1)$ when $\gamma=1$.

**Step 2 — Pool over time.** In an SNN the relevant population for channel $k$ is *all* of its values across the $T$ time steps and $B$ samples, because every one of them drives a spike decision. So the statistics are taken over $N=T\cdot B$ rather than $B$ [33]. This is the only change from BN that touches the statistics, and it is what distinguishes tdBN from per-step normalizations such as BNTT [49] (which keep separate statistics per time step).

**Step 3 — Scale to the threshold.** We want a *fixed, threshold-aware* fraction of neurons to fire regardless of depth. If $\hat{X}\sim N(0,1)$, then $\alpha V_{th}\hat{X}\sim N(0,(\alpha V_{th})^2)$, so the probability that a neuron exceeds $V_{th}$ is
$$P(\alpha V_{th}\hat X > V_{th}) = P\!\left(\hat X > \tfrac{1}{\alpha}\right) = 1-\Phi\!\left(\tfrac{1}{\alpha}\right),$$
which depends only on $\alpha$ — **not** on depth, $T$, or the raw input scale. Choosing $\alpha=1$ gives $1-\Phi(1)\approx 15.9\%$ of neurons above threshold per step, a healthy non-saturating firing regime [33].

**Step 4 — Gradient-norm preservation.** Substituting $Y=\alpha V_{th}\hat X$ into the Block Dynamical Isometry framework, Zheng et al. show the expected squared singular value of each block's input–output Jacobian is driven to $\approx 1$, so gradient norm is preserved through depth (no vanishing/exploding) [33]. The $\alpha V_{th}$ factor is exactly the scaling that makes this hold; replacing it with an arbitrary heuristic (e.g. a $1/\sqrt{T}$ factor) breaks the property and is *not* part of the paper.

**Backward pass.** With $s\equiv\alpha V_{th}$ constant, $Y=\gamma s\hat X+\beta$ gives, per channel,
$$\frac{\partial L}{\partial\beta}=\sum_i \frac{\partial L}{\partial Y_i},\qquad
\frac{\partial L}{\partial\gamma}=\sum_i \frac{\partial L}{\partial Y_i}\,s\,\hat X_i,\qquad
\frac{\partial L}{\partial\hat X_i}=\frac{\partial L}{\partial Y_i}\,\gamma s.$$
Propagating through the standardization (the usual BN input gradient, here pooled over $N=T\cdot B$):
$$\frac{\partial L}{\partial X_i}=\frac{1}{N\sqrt{\sigma^2+\varepsilon}}\left(N\frac{\partial L}{\partial\hat X_i}-\sum_j\frac{\partial L}{\partial\hat X_j}-\hat X_i\sum_j\frac{\partial L}{\partial\hat X_j}\hat X_j\right).$$
This is the formula implemented in `backward()` and verified against a central finite difference in the unit tests (`TdBNTest.InputGradientMatchesFiniteDifference`).

---

## Intuition

- **Why it works.** tdBN forces the membrane current entering each LIF layer to a known spread $\alpha V_{th}$. So no matter how deep the network is, every layer receives inputs whose magnitude is comparable to its threshold — neurons are neither all silent nor all saturated, and gradients flow [33].
- **How it modifies the membrane potential.** It re-centers (subtract $\mu$) and re-scales (divide by std, multiply by $\alpha V_{th}$) the current *before* it is integrated by the LIF neuron. It does **not** touch the recurrent membrane decay $\beta=e^{-\Delta t/(RC)}$; it only conditions the input current each step.
- **How it influences spikes.** By pinning the std to $\alpha V_{th}$, it fixes the *expected firing fraction* (≈16% at $\alpha=1$). Raising $\alpha$ pushes more neurons above threshold (denser spiking); lowering $\alpha$ sparsifies firing.

---

## Algorithm

**Pseudocode (per feature channel $k$, training mode):**

```
input  X        : (T*B, F) pre-activation (time-major)
params V_th, α, ε, γ_k, β_k
output Y        : (T*B, F)

μ_k   ← mean over all N = T*B rows of column k
σ²_k  ← variance over all N rows of column k          # pooled batch + time
running_mean_k ← (1-m)·running_mean_k + m·μ_k          # for inference
running_var_k  ← (1-m)·running_var_k  + m·σ²_k
for each row i:
    x̂  ← (X[i,k] - μ_k) / sqrt(σ²_k + ε)
    Y[i,k] ← γ_k · (α · V_th · x̂) + β_k                # β not scaled
# inference (eval) mode: use running_mean_k / running_var_k instead of batch μ,σ²
```

**Project implementation** ([`ThresholdDependentBatchNorm.hpp`](../../include/layers/spiking/ThresholdDependentBatchNorm.hpp)):

```cpp
const float scale = alpha * voltage_threshold;          // α·V_th
for (size_t f = 0; f < F; ++f) {
    // pooled mean/var over ALL total_rows = T*B (training); running stats (eval)
    float mean = ..., var = ...;                        // see forward()
    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (size_t r = 0; r < total_rows; ++r) {
        const float xn = (input.at(r, f) - mean) * inv_std;
        output.at(r, f) = gamma.at(0,f) * (scale * xn) + beta.at(0,f);  // β unscaled
    }
}
```

Training vs inference is toggled with `train(true)` / `train(false)`; in eval mode the layer normalizes with the accumulated `running_mean` / `running_var` so a single sample is processed deterministically. Because $\gamma,\beta$ are shared across time, at inference the whole affine map can be folded into the preceding `Linear` weights — tdBN therefore adds **zero** inference cost, unlike per-step schemes [50].

---

## Numerical example

One channel ($F=1$), $T=2$, $B=2$ → $N=4$ pooled values. Inputs (rows = $t_0b_0, t_0b_1, t_1b_0, t_1b_1$):
$$X = [\,0,\ 2,\ 4,\ 6\,],\qquad \gamma=1,\ \beta=0,\ \varepsilon=10^{-5}.$$
Note the two time steps have different per-step means (1 and 5); tdBN pools them.

1. **Mean:** $\mu = (0+2+4+6)/4 = 3$.
2. **Variance:** $\sigma^2 = \big[(0-3)^2+(2-3)^2+(4-3)^2+(6-3)^2\big]/4 = (9+1+1+9)/4 = 5$.
3. **Inverse std:** $1/\sqrt{5+10^{-5}} \approx 0.447214$.
4. **Normalization:** $\hat X = (X-3)\cdot 0.447214 = [-1.34164,\ -0.44721,\ 0.44721,\ 1.34164]$ (zero mean, unit variance).
5. **Threshold scaling**, with $\alpha=1$:
   - $V_{th}=1 \Rightarrow$ scale $=1$: $Y = [-1.3416,\ -0.4472,\ 0.4472,\ 1.3416]$, i.e. $N(0,1)$.
   - $V_{th}=2 \Rightarrow$ scale $=2$: $Y = [-2.6833,\ -0.8944,\ 0.8944,\ 2.6833]$, i.e. $N(0,4)=N(0,(\alpha V_{th})^2)$.
6. **Output** (the $V_{th}=2$ case is what a downstream LIF with threshold 2 receives): its standard deviation is exactly $\alpha V_{th}=2$, so ≈16% of values exceed the threshold of 2 — the balanced firing regime.

These exact numbers are asserted in `TdBNTest.PoolsStatisticsOverBatchAndTime` and `TdBNTest.ForwardOutputScaledByTdbnFactor`.

---

## Comparison

| Method | Statistics pooled over | Output target | Per-time-step params? | Foldable at inference? | Notes |
|---|---|---|---|---|---|
| **Batch Norm** (vanilla) | batch only | $N(0,1)$ | no | yes | threshold-agnostic; ignores time [56] |
| **Layer Norm** | features of one sample | $N(0,1)$ per sample | n/a | no | no batch coupling; rarely used in SNNs |
| **NeuNorm** [48] | channel dimension | data-normalized | no | partial | early SNN normalization, precursor to tdBN |
| **BNTT** [49] | batch, **separately per step** | per-step | **yes** ($\gamma_t,\beta_t$) | no | adapts to instantaneous dynamics; extra inference cost |
| **TEBN** [50] | batch+time, **re-weighted per step** | per-step weighted | yes (weights) | no | best at very low $T$; cannot fold into weights |
| **MPBN** [51] | membrane potential, after update | normalized $V$ | yes | partial | normalizes $V$ *after* the update, before firing |
| **tdBN** [33] | **batch + time (shared)** | $N(0,(\alpha V_{th})^2)$ | **no** (shared $\gamma,\beta$) | **yes** | threshold-aware; deep-SNN gradient-norm preservation; zero inference overhead |

**Advantages of tdBN** [33]: trains very deep SNNs directly; threshold-aware firing balance; shared parameters fold into weights → no inference overhead.
**Limitations** [49], [50]: a single shared scale/shift per channel is less expressive than per-time-step schemes; at very small $T$, TEBN can outperform it; it assumes a meaningful batch+time population (large enough $N$).

---

## Relationship with other parts of the project

- **LIF** ([Membrane Dynamics](Membrane-Dynamics.md)): tdBN conditions the current fed to each `LifBPTT` layer and is tied to its $V_{th}$. It is inserted **after every `Linear` and before the LIF** it feeds.
- **BPTT** ([SNN and Surrogate Gradients](SNN-and-Surrogate-Gradients.md)): tdBN's backward sits inside the unrolled BPTT graph — its input gradient is propagated together with the LIF gradients each step.
- **Surrogate gradients**: by keeping pre-activations near $V_{th}$, tdBN keeps the surrogate derivative in its active region, so gradients do not vanish across depth.
- **No-Spike Problem / [Spike-Rate Regularization](Spike-Rate-Regularization.md)**: tdBN structurally guarantees a non-zero firing fraction; the firing-rate band penalty is a complementary, loss-side guard against dead/bursting neurons.
- **[Weight Initialisation](Weight-Initialisation.md)**: tdBN reduces sensitivity to the initial weight scale (it re-standardizes each layer), but Kaiming-SNN init is still used so the very first forward pass is well-conditioned.
- **Loss function**: tdBN is loss-agnostic; in Experiment 05 it is combined with cross-entropy on the time-averaged readout.
- **Spike autoencoders**: the same layer can normalize the encoder/decoder currents of a spiking autoencoder; it composes with the spike-count loss and its rate regularizer.

---

## Usage (Experiment 05)

```json
{
  "training": {
    "batch_normalization": "threshold-dependent",
    "tdbn_alpha": 1.0
  }
}
```

When enabled, the E05 deep spiking classifier inserts a tdBN layer after each `Linear` and before each `LifBPTT` (`fc_in → tdBN → lif_in → (hidden_fc → tdBN → hidden_lif)* → fc_out`). The flag is ignored by the non-spiking `rnn` classifier. See [Experiment05](../Experiments/Experiment05.md).

---

## References

[33] Y. Zheng, H. Wu, G. Li, L. Deng, and Y. Xie, "Going deeper with directly-trained larger spiking neural networks," in *Proc. 35th AAAI Conf. Artificial Intelligence (AAAI)*, 2021, pp. 11062–11070. (tdBN; arXiv:2011.05280)

[48] Y. Wu, L. Deng, G. Li, J. Zhu, Y. Xie, and L. Shi, "Direct training for spiking neural networks: Faster, larger, better," in *Proc. 33rd AAAI Conf. Artificial Intelligence (AAAI)*, 2019, pp. 1311–1318. (NeuNorm)

[49] Y. Kim and P. Panda, "Revisiting batch normalization for training low-latency deep spiking neural networks from scratch," *Frontiers in Neuroscience*, vol. 15, 2021. (BNTT)

[50] C. Duan, J. Ding, S. Chen, Z. Yu, and T. Huang, "Temporal effective batch normalization in spiking neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. (TEBN)

[51] Y. Guo et al., "Membrane potential batch normalization for spiking neural networks," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023. (MPBN; arXiv:2308.08359)

[56] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, 2015, pp. 448-456.
