# Optimizers

The nn library provides optimization algorithms for training neural networks, including Adam and SGD.

## Theoretical Background

Neural network training minimizes a loss function $L(\theta)$ where $\theta$ represents the model parameters. Optimizers update parameters using gradients:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)$$

### Adam (Adaptive Moment Estimation)

Adam maintains per-parameter momentum and adaptive learning rates [2]:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

With bias correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update rule:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### SGD (Stochastic Gradient Descent)

Simple gradient descent with momentum:
$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

## Available Optimizers

| Token (`optimizer_type`) | Class | Reference default lr | State/param | Notes |
|---|---|---|---|---|
| `adam` (default) | `Adam` | 1e-3 | 2 (m, v) | AdamW when `weight_decay > 0` |
| `sgd` | `SGD` | — (0.01) | 1 (velocity) | Polyak momentum; SGDW when `weight_decay > 0` |
| `lion` | `Lion` | 1e-4 | **1** (momentum) | Sign-based update; half Adam's optimizer memory |
| `schedule-free-adamw` | `ScheduleFreeAdamW` | 2.5e-3 | 3 (x, z, v) | No lr schedule needed; train/eval iterates differ |

Built by `OptimizerFactory` from `TrainerConfig::optimizer_type`; in Experiment05 this is set
per profile via `training.optimizer_type`. An unknown token throws rather than falling back.

> **⚠️ A single lr is not comparable across these.** Their reference defaults span 1e-3 (Adam)
> to 1e-4 (Lion). Lion's update is `±lr` for *every* coordinate regardless of gradient
> magnitude, which is why its usable lr is much smaller. Any optimizer ablation must tune lr
> **per optimizer** or it measures the lr choice, not the optimizer.

### Per-optimizer learning rate

`reference_learning_rate(token)` (OptimizerFactory.hpp) is the single source of truth for each
optimizer's published default. Experiment05's `training.learning_rate` is **optional**: omit it
and `E05Config::Training::effective_learning_rate()` resolves it from the chosen optimizer, so
naming an optimizer without naming a rate still trains at a rate that suits *that* optimizer
rather than silently inheriting Adam's. An explicit value still wins, so sweeping lr works.

The run summary records the resolved `learning_rate` **and** a `learning_rate_source`
(`"profile"` or `"optimizer_default"`), so a result file says what it actually trained with —
the gap that made fixme.md D3 possible, where published numbers came from an effective lr 10×
below what the profiles declared, with nothing on disk recording it. Guarded by
`E05OptimizerLearningRate.*` and by a per-profile assertion in `e05_profile_audit_gtest`.

**Not implemented, deliberately** (fixme.md D5 records the full reasoning):
- **Sophia** — its diagonal-Hessian estimate needs a *second backward pass with resampled
  labels* (Gauss-Newton-Bartlett), which `Optimizer::step(span<Tensor*>)` cannot trigger: it
  receives only parameters and their gradients. Implementing it without that pass would make
  the "Hessian" equal to squared real-loss gradients — effectively Adam's second moment under
  a different name.
- **SOAP** — needs `eigh` of the Shampoo preconditioner, which does not exist in the
  backend-agnostic `Tensor` interface; adding it means extending `TensorBackendParityContract`
  across all four backends (XTensor, OpenCL, SYCL, Device).
- **Muon** — was implemented and ground-truthed against `muon-optimizer`, then removed on the
  author's instruction. It orthogonalizes 2-D matrices only and falls back to Adam for
  everything else, so it could never touch this project's 1×1 biophysical params (R, C, V_th);
  combined with its LLM-pretraining design target, it had no plausible role in the thesis
  ablation. Recoverable from git history if ever needed.

### Ground truth vs the reference implementations

Every optimizer is checked element-by-element against its **authors' own implementation**:
`scripts/testing/gen_optimizer_refs.py` drives the reference over a fixed parameter/gradient
sequence and records the parameter after each step into the committed
`src/core/optimizers/tests/fixtures/optimizer_refs.npz`; `optimizer_parity_gtest` replays the
identical data through ours. CI needs no Python (same pattern as the
[PyTorch layer parity](../Guides/Ground-Truth-and-Smoke-Testing.md) fixtures).

| Ours | Reference |
|---|---|
| `Adam`, `Adam` + `weight_decay`, `SGD` | `torch.optim.Adam` / `torch.optim.AdamW` / `torch.optim.SGD` |
| `Lion` | `lion-pytorch` |
| `ScheduleFreeAdamW` | `schedulefree` (`AdamWScheduleFreeReference`) |

This is not ceremony — it found two real bugs in code that had passed unit tests for months:
Adam applied its decoupled weight decay *after* the gradient step (AdamW defines it against
θ_{t-1}; torch agrees — decay-before reproduces torch with 0.0 error), and moving it earlier
exposed that assigning to a parameter **drops its gradient buffer** (`nn::Tensor` is a value
type). Both are fixed and pinned.

One deviation from upstream is deliberate and encoded in the fixture generator:
- **weight_decay scope** — upstream Lion/Schedule-Free decay every parameter; this project
  decays only 2-D weight matrices so SNN scalars keep τ=R·C intact. Every decay fixture uses a
  2-D parameter, where the two agree exactly.

## How It Is Implemented Here

### Adam — Standard Usage

```cpp
// File: include/optimizers/Adam.hpp
struct Adam : public Optimizer
{
    float learning_rate;
    float decay_rate_moment1 = 0.9F;
    float decay_rate_moment2 = 0.999F;
    float epsilon = 1e-8F;

    std::vector<nn::Tensor> moment1;  // per-param first moment
    std::vector<nn::Tensor> moment2;  // per-param second moment

    explicit Adam(float learning_rate_ = 0.001F, ...);

    void attach(std::span<nn::Tensor*> params) override;
    void step(std::span<nn::Tensor*> params) override;
    void zero_grad(std::span<nn::Tensor*> params) override;
};
```

### Per-Parameter-Group Learning Rates

SNN biophysical parameters (R, C, V_th) require a much smaller learning rate than
weight matrices, since large updates can push them into the `1e-6` clamp region
(see [Membrane-Dynamics](../Concepts/Membrane-Dynamics.md)) and destabilize $\tau=R\cdot C$
or the spike threshold. This project's own empirical default is a 10× reduction:
lr_SNN ≈ 1e-4 when global lr = 1e-3 (`TrainerConfig::snn_lr_scale = 0.1`) — an internal
engineering heuristic, not a literature-sourced figure (audit m-3: the citation
previously attached to this claim could not be verified and was removed).
Use `attach_with_scales()` to set per-parameter lr multipliers:

```cpp
// File: include/optimizers/Optimizer.hpp — base class, virtual
virtual auto attach_with_scales(std::span<nn::Tensor*> params,
                                std::span<const float> lr_scales) -> void;
```

`attach_with_scales()`, the `lr_scales_` storage it fills, and `weight_decay` all live on
the **`Optimizer` base** (fixme.md D5), so a caller holding an `Optimizer&`/`unique_ptr<Optimizer>`
— e.g. [`Trainer`](./Training.md), which builds its optimizer via `OptimizerFactory` — can
configure per-group learning rates without knowing the concrete type. Every optimizer in the
project (`Adam`, `SGD`, `SGDMinimal`) reads `lr_scales_` in its `step()`, computing
$\eta_i = \eta_\text{global} \times \texttt{lr\_scales\_}[i]$ (defaulting to 1.0 past the
vector's end). The base `attach_with_scales()` calls the virtual `attach()` (so concrete
per-parameter state — Adam's moments, SGD's velocity — is still allocated) and then stores the
scales; concrete optimizers do not override it.

> **Base `attach()` stores its params.** `Optimizer::attach()` records `attached_params_`
> (backing the no-arg `step()`/`zero_grad()`) and clears `lr_scales_`. Overrides must call
> `Optimizer::attach(params)` first. It used to be a literal no-op while its own comment
> claimed otherwise — so `SGD`, which relied on it, left `attached_params_` empty and its
> no-arg `step()` always threw.

The effective learning rate for parameter $i$ is:
$$\eta_i = \eta_\text{global} \times \text{lr\_scales}[i]$$

Example — separate lr for SNN layers vs weight matrices:
```cpp
Adam optimizer(/*lr=*/0.001F);

// Collect all params from mixed ANN+SNN model
auto all_params = model.params();  // e.g., [W1, W2, R_snn, C_snn, Vth_snn]

// Scale: full lr for weights, 0.1× for SNN biophysical params
std::vector<float> scales = {1.0f, 1.0f, 0.1f, 0.1f, 0.1f};
optimizer.attach_with_scales(all_params, scales);

// Training loop: step() uses per-param lr automatically
optimizer.step(all_params);
```

`TrainerConfig::snn_lr_scale` (default 0.1) sets the scale; the caller populates the
`scales` vector. [`Trainer`](./Training.md) does this for you, assigning the scale **only to
parameters with `size() == 1`** — R, C and V_th are always 1×1 tensors, while no weight or
bias ever is. Filling the vector uniformly instead (the pre-fix behavior) silently turned
`snn_lr_scale` into a *global* lr multiplier that throttled the whole network 10× — see
fixme.md D3.

### Decoupled Weight Decay (AdamW / SGDW)

`Optimizer::weight_decay` (default `0.0F`, disabled — on the base, so it is settable through
an `Optimizer&`) applies L2 regularization in the **decoupled** form of Loshchilov & Hutter
(ICLR 2019), which defines both the AdamW and SGDW variants. All three optimizers implement
it. After the standard update, each parameter is shrunk multiplicatively:

$$\theta_i \leftarrow \theta_i - \eta_i \cdot \lambda_\text{wd} \cdot \theta_i$$

Decoupling matters because folding the penalty into the gradient (classic L2)
makes Adam's per-coordinate $1/\sqrt{\hat v}$ scaling distort the effective decay
per weight; decoupled decay keeps a uniform shrink.

**Shape guard** — decay is applied **only to 2-D weight matrices**
(`rows > 1 && cols > 1`). Biases (`N×1`) and SNN biophysical scalars
(`1×1`: R, C, V_th) are skipped, so the membrane time constant $\tau = R\cdot C$
and the spike threshold are never pulled toward zero.

```cpp
// File: include/optimizers/Adam.hpp — inside step()
if (weight_decay > 0.0f && param.rows() > 1 && param.cols() > 1)
    param = param.add(param.multiply_scalar(-lr_i * weight_decay));
```

Wired from `TrainerConfig::weight_decay` in the `Trainer` constructor
(`optimizer_.weight_decay = cfg_.weight_decay;`). For Experiment05 it is set from
`training.weight_decay` in the profile JSON.

### Fused Backend Step (2026-07-15)

`Adam::step()` dispatches to a backend-fused kernel when the backend provides
one, via the concept-guarded template helper `try_fused_step()` (a template
because `if constexpr` only discards branches during template instantiation).
`OpenCLTensorBackend::adam_step_inplace()` runs the whole per-parameter update
(moment EMAs, bias correction, parameter step) as **one** `adam_step_kernel`
launch on device-resident buffers — the kernel had existed in
`KernelManager.cpp` but was never wired. The generic ~15-tensor-op path
remains for XTensor/Device/SYCL and as runtime fallback; decoupled weight
decay is layered identically on both paths. Gradients enter through
`grad_ref()` (no copy) instead of `param.grad()` (which copies and, on
OpenCL, forced a device→host→device round-trip per parameter per batch).

The `Trainer` batch loop's OpenCL `BatchScope` now also covers `zero_grad`,
gradient clipping, and `optimizer_.step()` — previously the optimizer ran
outside it, so each of its per-parameter kernels paid a full `clFinish`
(~250 per batch on the SNN-AE).

## Data Flow

```mermaid
flowchart LR
    subgraph Parameters
        theta[θ parameters]
    end

    subgraph Compute
        grad[∇L gradient]
        update[Parameter Update]
    end

    subgraph State
        m[moment]
        v[velocity]
    end

    theta --> grad
    grad --> update
    grad --> m
    m --> update
    grad --> v
    v --> update
    update --> theta
```

## Usage Example

```cpp
// File: include/optimizers/Optimizer.hpp
#include "nn/optimizers/Adam.hpp"

// Create optimizer with learning rate
nn::optimizers::Adam optimizer(0.001f, 0.9f, 0.999f, 1e-8f);

// Attach to model parameters
optimizer.attach(model.params());

// Training loop
for (int epoch = 0; epoch < epochs; ++epoch)
{
    for (auto& batch : dataloader)
    {
        optimizer.zero_grad();      // Clear previous gradients
        auto output = model.forward(batch, true);
        auto loss = criterion(output, target);
        model.backward(loss.grad());
        optimizer.step(model.params());  // Update weights
    }
}
```

## Common Pitfalls

1. **Learning Rate**: Too high causes divergence, too low causes slow convergence

2. **Gradient Clipping**: Use `grad_clip_norm` to prevent exploding gradients in deep networks

3. **Momentum**: High momentum can cause oscillations; start with defaults (0.9)

4. **Adam epsilon**: Default (1e-8) prevents division by zero; too large slows learning

## See Also

- [Tensor](./Tensor.md) — Gradient computation
- [Layers](./Layers.md) — Forward/backward passes
- [Adam Optimiser](../Concepts/Adam-Optimiser.md) — Detailed Adam explanation
- [Training](./Training.md) — `TrainerConfig.snn_lr_scale` field
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — SNN parameter landscape

## References

[1] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. on Learning Representations (ICLR)*, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[2] S. Ruder, "An overview of gradient descent optimization algorithms," arXiv preprint arXiv:1609.04747, 2016. [Online]. Available: https://arxiv.org/abs/1609.04747
