# Layers

A neural network is built by stacking small, reusable building blocks called
**layers**. Each layer takes a batch of numbers in, transforms them in some
fixed way, and passes numbers out to the next layer. This page catalogues
every layer type the `nn` library provides — dense (fully-connected), spiking,
convolutional, residual, recurrent (LSTM, GRU), and attention (LayerNorm,
multi-head self-attention, sinusoidal positional encoding, Transformer encoder
block) — and shows how to combine them.

If you are new to neural networks, read this alongside
[Autoencoders](../Concepts/Autoencoders.md) for the bigger picture of how
layers compose into a trainable model.

## Theoretical Background

### What a layer actually computes

The most common layer, the **dense** (or "fully-connected", or "linear")
layer, computes:

$$y = f(Wx + b)$$

In plain terms: take the input numbers $x$ (say, 128 of them), multiply each
one by a set of learned weights $W$ and add them up (that's $Wx$), add a
learned offset $b$ (the "bias"), and finally squash the result through a
non-linear function $f$ (the "activation function"). Multiple such layers
stacked in sequence is what makes a "deep" network.

Why the non-linearity $f$ matters: stacking two purely linear layers
($y = W_2(W_1 x)$) is mathematically identical to *one* linear layer — you gain
nothing. The activation function is what lets the network represent curves and
decision boundaries instead of only straight lines/planes.

### Activation functions

An activation function decides how strongly, and in what shape, a neuron's
combined input gets passed forward. Common choices:

- **ReLU**: $f(x) = \max(0, x)$ [4] — passes positive values through unchanged
  and zeroes out negative ones. Cheap to compute, and the default choice for
  most dense/convolutional networks.
- **LeakyReLU**: $f(x) = x$ if $x > 0$, else $\alpha x$ — like ReLU, but lets a
  small fraction ($\alpha$, e.g. 0.01) of negative values through instead of
  zeroing them. This avoids a failure mode called a "dead neuron", where a
  neuron gets stuck always outputting zero and its weights stop receiving any
  gradient to learn from.
- **Sigmoid**: $f(x) = 1/(1 + e^{-x})$ — squashes any real number into the
  range $(0, 1)$. Useful when the output should be interpreted as a
  probability or a gate that's "on" or "off" (see the LSTM gates below).
- **Tanh**: $f(x) = \tanh(x)$ — like sigmoid but squashes into $(-1, 1)$,
  centred at zero.

#### Fast activation approximations

Computing $e^x$ (needed by sigmoid and tanh) is one of the more expensive
single operations a CPU can do — expensive enough that, in a tight loop like an
LSTM running over hundreds of time steps, it shows up as a measurable fraction
of total training time. `include/layers/activations/FastActivations.hpp`
provides cheaper approximations that use only a division, trading a small,
bounded amount of accuracy for speed:

| Function | Formula | Max error | Use case |
|---|---|---|---|
| `sigmoid_fast(x)` | $0.5 + x / (2(1+\|x\|))$ | large — not a close approximation (see below) | opt-in speed/fidelity trade only |
| `tanh_fast(x)` | $x / (1 + \|x\|)$ | reaches 0.306 on [-4,4] (x=2: tanh=0.964 vs tanh_fast=0.667) | opt-in speed/fidelity trade only |

**These are not close approximations of the real functions**, and are NOT the
default. A `sigmoid_exact_block`/`tanh_exact_block`/`tanh_exact_tensor` family
computes the real (`std::exp`/`std::tanh`-based) values, and dispatcher
functions `sigmoid_block(..., bool exact)` / `tanh_block(..., bool exact)` /
`tanh_tensor(x, bool exact)` pick fidelity at runtime from a single flag —
`LSTMLayerImpl::exact_activations` defaults to `true` (exact), since
PyTorch/snnTorch is this project's correctness reference. The fast forms are
an explicit opt-in trade via `ThesisConfig::Numerics::exact_activations`.

Independent of fast-vs-exact, the "block" variants below go one step further
than the plain `*_tensor` ones: instead of first copying out a slice of a
larger tensor and *then* applying the activation to the copy (two passes over
memory, one allocation), they read the slice and apply the activation in a
single pass:

```cpp
// Reads pre[:,col_start:col_start+gate_size] and applies sigmoid in one pass.
// Avoids one alloc + one read-scan vs. sigmoid_fast_tensor(pre.block(...)).
nn::activations::sigmoid_exact_block(pre, col_start, gate_size);  // default (exact) fidelity
nn::activations::sigmoid_fast_block(pre, col_start, gate_size);   // opt-in fast approximation
```

**Measured impact** (batch size 1, input 128, hidden 32): before this fusion,
computing the LSTM's gates took 43.7% of the time spent on one time step; after,
it takes 1.1%, and a full time step runs 2.85× faster overall. See
[LSTM-and-BPTT](../Concepts/LSTM-and-BPTT.md#performance-characteristics-and-optimizations)
for the full measurement.

**Works with every tensor backend.** As of 2026-07-15, all four functions
above are written generically over the tensor backend type, rather than being
hard-wired to one specific backend. Before that, they silently compiled
correctly only for whichever single backend a given build happened to select —
nothing exercised them against any other backend, so a break would have gone
unnoticed. `pytorch_parity_gtest` (see
[Ground-Truth-and-Smoke-Testing](../Guides/Ground-Truth-and-Smoke-Testing.md))
now runs the LSTM layer against every backend side-by-side against the same
PyTorch reference to catch this class of bug.

### Convolutional layers

A convolutional layer applies the same small filter at every position of its
input, rather than a fully separate weight for every input/output pair. This
is what a "dense" layer would need to give up spatial structure entirely (it
treats the input as one long list of numbers); a convolution instead slides a
small pattern-detector across the input and asks "does this pattern appear
here?" at every position — which is why convolutions are the standard choice
for images, and also useful for 1D signals like audio:

$$y_{i,j,k} = \sum_{m,n} x_{i,m,n} \cdot w_{k,m,n} + b_k$$

For a 1D convolution, the output length depends on the input length $L$, the
padding $P$, the filter (kernel) size $K$, and the stride $S$ (how far the
filter moves between applications):

$$L_{out} = \lfloor(L + 2P - K)/S\rfloor + 1$$

**Implementation status:** `Conv1dImpl` (declared in
`include/layers/convolution/Conv1d.hpp`, implemented in
`src/core/layers/convolution/Conv1d_impl.cpp`) is a real im2col/col2im
convolution with full backward, not a placeholder — weights shape
`(C_in*K, C_out)`, He-initialised. `MaxPool1dImpl` / `MaxPool2dImpl` (headers
`convolution/MaxPool1d.hpp`, `MaxPool2d.hpp`) are likewise real, with argmax
routing in the backward pass. All three are exercised end-to-end (known-value
and shape tests) by `fundamental_mechanisms_convolution_gtest`.

## How It Is Implemented Here

Every layer in this codebase inherits from a common base class, `nn::Module`,
which fixes the contract every layer must satisfy: it must be able to run
forward (compute an output from an input) and backward (compute how the loss
would change if its inputs changed slightly — the gradient), and it must be
able to report which of its internal numbers are trainable weights:

```cpp
// File: include/layers/base/Module.hpp
template <typename Backend>
struct Module
{
    using Tensor = nn::TensorImpl<Backend>;

    virtual auto forward(const Tensor& input, bool requires_grad = true) -> Tensor = 0;
    virtual auto backward(const Tensor& grad_output) -> Tensor = 0;   // returns grad w.r.t. input
    virtual auto params() -> std::span<Tensor*> { return {}; }        // trainable-param pointers
    virtual void train(bool on) {}
    virtual void reset_state() {}    // clears persistent state (e.g. LIF membrane potential)
};
```

### Dense (Linear) layer

The simplest layer, implementing exactly the $y = f(Wx+b)$ formula above (here
without an activation — the activation is applied by a separate layer, or
fused in for speed where noted):

```cpp
// File: include/layers/dense/Linear.hpp
template <typename Backend>
struct LinearImpl : public Module<Backend>
{
    Tensor weight;   // (out_features, in_features) — allocated, NOT initialized here
    Tensor bias;     // (out_features, 1)

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // Real forward also validates in_features and flattens any leading
        // (batch/time) dimensions before this multiply — simplified here.
        Tensor result = input.matmul_transposed(weight);   // x @ weightᵀ
        result.add_col_vector_to_rows_inplace(bias);
        return result;
    }
};
```

Backward-path note (OpenCL backend only): computing the weight gradient
$dL/dW$ needs the transpose of the incoming gradient. Rather than materialising
that transpose as its own tensor and then multiplying, the OpenCL backend has
a fused `matmul_lhs_transposed` kernel that does the transpose and the multiply
in one step. This exists purely for speed — it produces the same numbers,
just faster — and is checked by the OpenCL backend tests plus timing rows in
`src/core/tensor/tests/tensor_perf_bench.cpp`.

### Spiking neuron (Leaky Integrate-and-Fire)

A LIF ("Leaky Integrate-and-Fire") neuron is a different kind of building
block: instead of producing a continuous number every time it's called, it
accumulates ("integrates") its input over time into a "membrane voltage" that
slowly decays ("leaks"), and emits a single spike (a 1, otherwise 0) whenever
that voltage crosses a threshold — much like a bucket that fills with water,
slowly drains, and tips over once full. See
[SNN and Surrogate Gradients — Plain Language Guide](../Concepts/Plain/SNN-and-Surrogate-Gradients.md)
for the full intuition and analogy, and
[SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) for
the equations.

`LifImpl` keeps that membrane voltage as persistent state across sequential
`forward()` calls (i.e. calling it in a loop, once per time step, is how you
run it over a sequence). Its trainable parameters are the resistance `R`,
capacitance `C`, and firing threshold `voltage_threshold` (V_th) — the
"electrical circuit" constants that control how quickly the neuron forgets and
how easily it fires. An optional "spike-frequency adaptation" mechanism
(`adapt_decay` / `adapt_coupling`) can temporarily raise the threshold after
each spike, so a neuron doesn't fire on every single input:

```cpp
// File: include/layers/spiking/Lif.hpp
template <typename Backend>
struct LifImpl : public Module<Backend>
{
    float delta_t = 1.0F;
    Tensor resistance, capacitance, voltage_threshold;  // trainable 1×1
    Tensor v_mem;           // persistent membrane state (B×F)
    float adapt_decay    = 0.9F;  // threshold decay factor
    float adapt_coupling = 0.0F;  // threshold rise per spike (0 = disabled)
    Tensor adapt_a;               // adaptation variable (B×F)

    // β = exp(-Δt/(R·C)); V[t] = β·V[t-1] + I[t]
    // Effective threshold: V_th + adapt_a
    // On spike: V → V_reset; adapt_a += adapt_coupling
};
```

`LifBPTTImpl` is the alternative you use when you want to train on a whole
sequence at once rather than one time step at a time. It unrolls the entire
sequence internally in a single `forward(input (T*B,F))` call (see
[Time-Major Layout](../Concepts/Time-Major-Layout.md) for what the `(T*B,F)`
shape means) and computes exact gradients for R, C, and V_th through the whole
sequence — this technique is called **Backpropagation Through Time (BPTT)**,
explained in [LSTM-and-BPTT](../Concepts/LSTM-and-BPTT.md). It can save and
reload its parameters via `state_dict`/`load_state_dict`, so a trained
network's learned R, C, and V_th survive being written to disk and loaded back.
Full detail: [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md).

> **Note:** R and C never need to be recovered individually — every equation
> in this layer only ever uses their product, the time constant
> $\tau = R \cdot C$. Two different (R, C) pairs with the same product behave
> identically. See [Membrane Dynamics](../Concepts/Membrane-Dynamics.md).

**Temporal classifier example (Experiment05).** `ThesisDsnnClassifier` stacks
`Linear → LifBPTT → … → Linear` and feeds a single static feature vector
repeatedly over `kSnnTimeSteps` (default 16) time steps — turning a
non-temporal input into a spike train by constant-current encoding — then reads
out the average spike rate over time as the class score. This is a genuine use
of the *temporal* dynamics of spiking neurons, not just a one-shot classifier
wearing an SNN costume. See [Experiment05](../Experiments/Thesis.md).

**OpenCL backend note.** When running on the OpenCL (GPU) backend, `LifImpl`
uses two fused GPU kernels instead of the generic tensor operations: one that
does the membrane update and spike generation together
(`lif_step_inplace`), and one for the backward-pass surrogate gradient
(`lif_grad`). Backends that don't provide these fall back to the plain,
generic implementation automatically — this is purely a speed optimisation,
never a behaviour change. Verified by `opencl_tensor_backend_lif_gtest`,
including `LeakyLayerForwardParityOnOpenCLBackend` and
`LeakyLayerBackwardExponentialSurrogateOnOpenCLBackend`.

### Threshold-Dependent Batch Normalization (tdBN)

Deep spiking networks have a stability problem: as a signal passes through
many LIF layers in sequence, the membrane voltages can drift — growing without
bound in some layers, shrinking to nothing in others — making training
unreliable. tdBN is a normalisation step, inserted between a dense layer and
its LIF layer, that rescales the incoming current so its statistics (mean and
spread) stay in the same, well-behaved range no matter how deep the network
is. Concretely, it normalises per-channel, pooling statistics over **both the
batch and the time dimension**, then rescales by $\alpha V_{th}$ so the LIF
layer downstream always receives input distributed as
$N(0,(\alpha V_{th})^2)$ [33]:

$$Y_k = \gamma_k(\alpha V_{th}\hat{X}_k) + \beta_k \qquad (\beta \text{ unscaled})$$

Full theory, derivation and a worked numeric example:
[Threshold-Dependent Batch Normalization](../Concepts/Threshold-Dependent-Batch-Normalization.md).

```cpp
// File: include/layers/spiking/ThresholdDependentBatchNorm.hpp
template <typename Backend>
class ThresholdDependentBatchNormImpl : public Module<Backend>
{
public:
    float alpha = 1.0F;              // α: target std = α·V_th (paper default 1)
    float voltage_threshold = 1.0F;  // V_th of the downstream LIF layer
    int time_steps = 1;              // T: statistics pool over batch AND time
    float eps = 1e-5F;
    float momentum = 0.1F;           // EMA rate for inference running stats
    Tensor gamma;   // learned per-channel scale (1×F)
    Tensor beta;    // learned per-channel shift (1×F)
    Tensor running_mean, running_var; // inference buffers (1×F)

    explicit ThresholdDependentBatchNormImpl(
        size_t num_features, float vth = 1.0F, int T = 1,
        float alpha_ = 1.0F, float eps_ = 1e-5F, float momentum_ = 0.1F);
};
```

Typical placement — between a `Linear` layer and its `LifBPTT`, in a deep SNN
encoder:
```cpp
ThresholdDependentBatchNormImpl<Backend> tdbn(64, /*vth=*/1.0f, /*T=*/10);
auto h = tdbn.forward(fc.forward(input, true), true);
```

### ResidualBlock vs ResNetBlock

A **residual (skip) connection** adds a layer's input directly to its output
($y = x + F(x)$ instead of just $y = F(x)$), which helps gradients flow
through very deep networks without vanishing. This project has two related
classes with different maturity levels:

| Class | File | Shape | Backward |
|---|---|---|---|
| `ResidualBlockImpl` | `residual/ResidualBlock.hpp` | Dense (Linear → ReLU → Linear + skip) | ✓ |
| `ResNetBlockImpl` | `residual/ResNetBlock.hpp` | Convolutional (Conv2d → ReLU → Conv2d + skip, both ReLUs) | ✓ |

Both are complete, instantiable implementations with full backward passes.
`ResidualBlockImpl` is the dense/MLP variant; `ResNetBlockImpl` is the
two-layer convolutional variant, using two separate `ReLU` instances (each
caches its own activation mask independently) and shape-aligning the skip
path (identity when shapes match, zero-padded otherwise) in both forward and
backward. See [Residual Blocks](../Concepts/Residual-Blocks.md).

### Poisson Latent Layer (SNN-VAE)

This layer is the spiking-network equivalent of the "reparameterisation trick"
used in a standard Variational Autoencoder (VAE): instead of sampling a
Gaussian latent variable, it samples spike counts from a Poisson process
[29, 30]. See [Autoencoders](../Concepts/Autoencoders.md) if you are unfamiliar
with what a VAE's latent space is for.

> **Bug fix (2026-05-01):** a sign error in the KL-divergence formula (a term
> in the loss that measures how far the learned distribution has drifted from
> a fixed target) meant the old code always computed a *negative* penalty,
> which rewarded the network for drifting further away from the target rather
> than staying close to it — the opposite of the intended effect. The
> corrected formula, $\text{KL}(\text{Poisson}(\lambda) \| \text{Poisson}(\lambda_0)) = \lambda_0 - \lambda + \lambda \log(\lambda/\lambda_0) \geq 0$,
> is now enforced by regression tests `PoissonLatentTest.KLExactKnownValue` and
> `PoissonLatentTest.KLZeroAtPrior`, in `fundamental_mechanisms_spiking_gtest`.

```cpp
// File: include/layers/spiking/PoissonLatentLayer.hpp
template <typename Backend>
class PoissonLatentLayerImpl : public Module<Backend>
{
public:
    int time_steps = 1;
    float prior_rate = 0.1F;  // λ₀ for KL divergence
    float beta_kl = 1.0F;    // β weighting

    float kl_loss() const;              // add β*kl_loss() to total loss
    const Tensor& last_rates() const;   // λ values for sparsity logging

    explicit PoissonLatentLayerImpl(int T = 1, float prior_rate = 0.1F, float beta_kl = 1.0F);
    // forward(train): λ=softplus(z); s~Poisson(λ·T); return s/T
    // forward(infer): return λ (no stochastic sampling)
};
```

### LSTM layer

An LSTM ("Long Short-Term Memory") is the classical (non-spiking) way to give
a network memory across a sequence: at every time step, three learned "gates"
decide what to forget from the previous state, what new information to add,
and what to output. See [LSTM-and-BPTT](../Concepts/LSTM-and-BPTT.md) for the
full gate equations and a from-scratch derivation of the training procedure.
This implementation's equations and backward pass are checked against
Hochreiter & Schmidhuber's original paper [5] and Greff et al.'s
comprehensive comparison of LSTM variants [6].

**Location:** `include/layers/lstm/LSTMLayer.hpp`

The three weight tensors stack all four gates (named i, f, o, g, following the
convention in [6]) into single matrices, so one matrix multiply computes all
four gates' pre-activations at once instead of four separate multiplies:
- `W_` : (4H × D) — input-to-hidden weights
- `U_` : (4H × H) — hidden-to-hidden (recurrent) weights
- `b_` : (4H × 1) — bias; the forget-gate's slice is initialised to 1 rather
  than 0, a well-known trick [7] that makes the network default to
  "remember everything" early in training rather than forgetting by default.

```cpp
// File: include/layers/lstm/LSTMLayer.hpp
// forward dispatches on input rank:
//   (T, D)    → (T, H)     single sequence; persists h0_/c0_ across calls
//   (B, T, D) → (B, T, H)  batch; each of B samples starts from zero state

nn::models::lstm::LSTMLayer layer(input_size, hidden_size);

layer.reset_state();                             // zero h0_, c0_
auto out = layer.forward(seq_td,  true);         // (T,D)   → (T,H)
auto dx  = layer.backward(grad_th);              // (T,H)   → (T,D)

auto out3 = layer.forward(seq_btd, true);        // (B,T,D) → (B,T,H)
auto dx3  = layer.backward(grad_bth);            // (B,T,H) → (B,T,D)

auto params = layer.params();                    // span<nn::Tensor*>: {W_, U_, b_}
```

**Performance optimisations applied** (see
[LSTM Performance Guide](../Guides/LSTM-Performance.md) for the measurements
behind each one):
- Gate activations use the fused `sigmoid_block` / `tanh_block` dispatchers
  described above (no intermediate copy) — exact by default
  (`exact_activations = true`), with the fast approximations available as an
  explicit opt-in trade.
- The bias transpose `b_T = b_.transpose()` is computed once before the time
  loop starts, instead of being recomputed on every step.
- Reading and writing one time step's slice uses vectorised `slice_time` /
  `setBlock` calls instead of a manual element-by-element loop.

### GRU layer

A **GRU** ("Gated Recurrent Unit" [72]) is a lighter alternative to the LSTM:
it keeps a single hidden state (no separate cell state) and uses two gates
instead of three. A *reset* gate decides how much of the previous state to
mix into the candidate update, and an *update* gate interpolates between the
old state and that candidate. Chung et al. [73] found GRUs match LSTMs on
sequence modelling with fewer parameters — which is exactly why the Meeting01
comparison reports GRU-AE as a first-class baseline next to LSTM-AE.

**Location:** `include/layers/gru/GRULayer.hpp`

Gate equations (cuDNN / PyTorch convention — the reset gate is applied *after*
the recurrent matrix multiply):

$$r_t = \sigma(x_t W_r^\top + h_{t-1} U_r^\top + b_r)$$
$$z_t = \sigma(x_t W_z^\top + h_{t-1} U_z^\top + b_z)$$
$$n_t = \tanh\!\big(x_t W_n^\top + b_n + r_t \odot (h_{t-1} U_n^\top)\big)$$
$$h_t = (1 - z_t)\odot n_t + z_t \odot h_{t-1}$$

The three gates $[r\,|\,z\,|\,n]$ are stacked into single matrices, so one
matmul computes all three pre-activations (same trick as the LSTM's four):
- `W_` : (3H × D) — input-to-hidden
- `U_` : (3H × H) — hidden-to-hidden (recurrent)
- `b_` : (3H × 1) — bias

Same 2-D `(T, D)` / 3-D `(B, T, D)` shape contract as `LSTMLayer`, same
`exact_activations` flag, batch gradient accumulation, and `reset_state()`
semantics. The backward pass is a full BPTT and is checked against central
finite differences on `W_`, `U_`, `b_`, and the input gradient
(`gru_layer_gtest.cpp`).

### LayerNorm

**Layer normalization** [75] standardises each row of the input independently
— subtract that row's mean, divide by its standard deviation over the feature
dimension, then apply a learned per-feature gain `gamma` and bias `beta`.
Unlike batch normalization it has no dependence on other samples in the batch
and no train/eval mode difference, which is why the Transformer uses it.

**Location:** `include/layers/normalization/LayerNorm.hpp`

$$\hat{x}_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \qquad
  y_{ij} = \gamma_j\,\hat{x}_{ij} + \beta_j$$

- `gamma` : (1 × D), initialised to 1
- `beta`  : (1 × D), initialised to 0

The backward pass uses the standard reduced form
`dx = (is/D)·(D·dxhat − Σ dxhat − xhat·Σ(dxhat·xhat))` and is finite-difference
checked on `dx`, `dgamma`, `dbeta` (`layernorm_gtest.cpp`). Calling `backward()`
before `forward(requires_grad=true)` throws (a `forward_cached_` flag guards it).

### Multi-head self-attention

**Self-attention** lets every position in a sequence look at every other
position and pull in a weighted combination of their values — the weights are
computed from the similarity of a *query* and a *key* [74]. "Multi-head" runs
several such attention operations in parallel on projected subspaces and
concatenates the results.

**Location:** `include/layers/attention/MultiHeadAttention.hpp`

For each head $h$ (with $d_k = d_\text{model}/n_\text{heads}$):

$$S_h = \frac{Q_h K_h^\top}{\sqrt{d_k}}, \qquad
  A_h = \operatorname{softmax_{rows}}(S_h), \qquad
  O_h = A_h V_h$$

then $\operatorname{concat}(O_1,\dots,O_H)$ is projected by $W_O$. Q, K, V, O
are four `LinearImpl` sub-modules, so their weights and biases (and gradients)
come for free from the existing dense layer. The softmax backward uses
`ds[i,j] = a[i,j]·(da[i,j] − ⟨da[i,:], a[i,:]⟩)`. Every projection parameter
and the input gradient are finite-difference checked
(`multi_head_attention_gtest.cpp`); attention weights are verified to sum to 1
per row. A head count that does not divide `d_model` throws (and the member
initialiser is SIGFPE-safe against it).

### Sinusoidal positional encoding

Self-attention is permutation-invariant, so position has to be injected
explicitly. `sinusoidal_positional_encoding<Backend>(seq_len, d_model)` [74] is
a **free function**, not a `Module` — it has no parameters and no backward
pass, it just returns a fixed `(seq_len, d_model)` tensor of sines and cosines
at geometrically-spaced frequencies, added to the embeddings. (It is excluded
from the auto-generated `Layers.hpp` aliases for exactly this reason — it is
not a `FooImpl<Backend>` class.)

**Location:** `include/layers/attention/PositionalEncoding.hpp`

### Transformer encoder block

One **post-norm** Transformer encoder block [74] composes the pieces above:

```
a   = MultiHeadAttention(x)
n1  = LayerNorm1(x + a)
f   = W2 · ReLU(W1 · n1 + b1) + b2      // position-wise feed-forward
out = LayerNorm2(n1 + f)
```

**Location:** `include/layers/attention/TransformerEncoderBlock.hpp`

Sub-modules: `mha_`, `ln1_`, `ff1_`, `relu_`, `ff2_`, `ln2_`. Each is already
individually gradient-checked; the block adds a *composed* finite-difference
check that perturbs one representative parameter from every sub-module plus the
input (`transformer_encoder_block_gtest.cpp`). Self-attention only — sufficient
for the encoder-only / bottlenecked Transformer autoencoder (see
[Models](./Models.md)).

### Spike losses

A loss function measures how wrong the network's output is, and that number
is what backpropagation minimises. SNN outputs need loss functions matched to
how the spikes encode information — see
[Spike Encoding](../Concepts/Spike-Encoding.md) for the distinction between
these two encodings:

| Class | File | Use case |
|---|---|---|
| `SpikeCountLossImpl` | `losses/SpikeCountLoss.hpp` | Rate-coded outputs (information is in *how many* spikes fired) — mean-squared error on spike counts, plus a regularisation term that discourages neurons from always firing or never firing |
| `SpikeTimeLossImpl` | `losses/SpikeTimeLoss.hpp` | Latency-coded outputs (information is in *when* the first spike fires) — mean-squared error on first-spike timing |

## Data Flow

```mermaid
flowchart TB
    subgraph Input
        x[Input Tensor<br/>batch×input_dim]
    end

    subgraph Layer
        weights[Weight Matrix<br/>input_dim×output_dim]
        bias[Bias Vector<br/>1×output_dim]
        act[Activation]
    end

    subgraph Output
        y[Output Tensor<br/>batch×output_dim]
    end

    x --> weights
    weights --> act
    bias --> act
    act --> y
```

## Usage Example

```cpp
// File: include/layers/Layers.hpp
#include "layers/dense/Linear.hpp"
#include "layers/activations/ReLU.hpp"

// Create a simple MLP: 128 -> 64 -> 32
nn::Linear fc1(128, 64);
nn::ReLU relu1;
nn::Linear fc2(64, 32);

// Forward pass
nn::Tensor x = /* input data */;
nn::Tensor h = fc1.forward(x, true);
h = relu1.forward(h, true);
nn::Tensor y = fc2.forward(h, true);
```

## Common Pitfalls

1. **Shape mismatch.** A layer's input dimension must equal the previous
   layer's output dimension — if layer A outputs 64 numbers per sample, layer
   B must expect exactly 64 numbers in.

2. **Forgetting to zero gradients.** Gradients from `backward()` *accumulate*
   into each parameter by default — they don't automatically reset before the
   next batch. Always call `optimizer.zero_grad()` before computing a new
   `backward()` pass, or gradients from old batches will keep piling onto the
   new ones.

3. **Spiking neuron state not reset.** LIF layers keep their membrane voltage
   between calls, on purpose, so they can process a sequence one step at a
   time. But that means starting a *new, independent* sequence without
   calling `reset_state()` first will let the old sequence's leftover voltage
   leak into the new one.

4. **Weight initialisation.** Starting all weights at the same value, or at
   values too large/small, makes gradients vanish or explode as they pass
   through many layers. Use one of the provided initialisers (Xavier or
   Kaiming — see [Weight Initialisation](../Concepts/Weight-Initialisation.md))
   rather than ad-hoc random values.

## See Also

- [Tensor](./Tensor.md) — the data structure every layer operates on
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — LIF neuron, tdBN, PoissonLatent, in depth
- [Spike Rate Regularization](../Concepts/Spike-Rate-Regularization.md) — SpikeCountLoss with dead/burst prevention
- [Spike Encoding](../Concepts/Spike-Encoding.md) — rate vs latency coding; SpikeTimeLoss
- [Residual Blocks](../Concepts/Residual-Blocks.md) — skip connections, explained
- [Weight Initialisation](../Concepts/Weight-Initialisation.md) — why initial weight values matter

## References

[1] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in *Proc. 13th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, 2010, pp. 249–256.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2015. [Online]. Available: https://arxiv.org/abs/1502.01852

[29] K. Kamata et al., "Fully spiking variational autoencoder," in *Proc. AAAI Conf. Artificial Intelligence*, 2022.

[30] C. Chen et al., "ESVAE: An efficient spiking variational autoencoder with reparameterizable Poisson spiking sampling," arXiv:2310.14839, 2024.

[5] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. doi: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

[6] K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink, and J. Schmidhuber, "LSTM: A search space odyssey," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 28, no. 10, pp. 2222–2232, 2017. arXiv: [1503.04069](https://arxiv.org/abs/1503.04069)

[7] R. Jozefowicz, W. Zaremba, and I. Sutskever, "An empirical evaluation of recurrent network architectures," in *Proc. ICML*, 2015, pp. 2342–2350.

[33] Y. Zheng et al., "Going deeper with directly-trained larger spiking neural networks," in *Proc. AAAI Conf. Artificial Intelligence*, 2021. [Online]. Available: https://arxiv.org/abs/2011.05280

[72] K. Cho et al., "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. EMNLP*, 2014, pp. 1724–1734. arXiv: [1406.1078](https://arxiv.org/abs/1406.1078)

[73] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," *NeurIPS Deep Learning Workshop*, 2014. arXiv: [1412.3555](https://arxiv.org/abs/1412.3555)

[74] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008. arXiv: [1706.03762](https://arxiv.org/abs/1706.03762)

[75] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," *arXiv:1607.06450*, 2016. [Online]. Available: https://arxiv.org/abs/1607.06450

> In-text numbers follow the project-wide numbering in [References](../References.md). The entries cited above are reproduced here.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2015, pp. 1026–1034. [Online]. Available: https://arxiv.org/abs/1502.01852
