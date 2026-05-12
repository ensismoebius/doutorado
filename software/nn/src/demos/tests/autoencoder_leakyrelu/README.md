# Spiking Autoencoder (LifBPTT) — Experiment Guide

This document explains the experiment implemented in `src/demos/autoEncoderLeakyReLUAndSpikeTest/autoEncoderLeakyReLUAndSpikeTest.cpp`.

It is designed to stand alone as educational technical documentation: you should be able to understand what the demo does, why it is structured this way, what behavior to expect, and which parameters you can safely change.

> Note on naming: despite the filename containing “LeakyReLU”, this demo **does not** use the Lif ReLU activation. The “leaky” here refers to **leaky integrate-and-fire (LIF)** neuron dynamics implemented by `LifBPTT`.

---

## 1. What this experiment is (and what it is not)

**Goal (experimental intent):** validate that a deep spiking autoencoder can be trained end-to-end in this C++ framework using:

- a time-unrolled spiking layer (`LifBPTT`) with **full BPTT** (backpropagation through time),
- an explicit **surrogate gradient** for spike non-differentiability,
- stable optimization practices (weight initialization + gradient clipping),
- deterministic, synthetic data.

**Not the goal:** benchmarking accuracy on real datasets, learning rich latent structure, or demonstrating state-of-the-art encoding. The input is intentionally simple so that failures are interpretable.

**Observable outputs:**

- Console prints including loss.
- A training log written to `cpp_loss_log.txt` with `epoch,loss`.

---

## 2. Core definitions (terms used throughout)

- **Autoencoder:** a model trained to reconstruct its input $x$ as output $\hat{x}$. It learns a compressed representation (latent code) in the middle.
- **Latent / bottleneck:** the smallest representation $z$ (lower dimensionality), forcing compression.
- **Spike train:** a sequence of events in time, typically $S[t] \in \{0,1\}$.
- **LIF neuron (leaky integrate-and-fire):** a neuron with a membrane potential $V$ that decays over time (“leaks”), integrates input current, and emits a spike when $V$ crosses a threshold.
- **BPTT (backpropagation through time):** gradient computation for dynamical systems/RNNs by unrolling the recurrence across time steps and applying the chain rule backward.
- **Surrogate gradient:** a differentiable approximation used in the backward pass for the derivative of the spike function.
- **Readout mode:** using the membrane potential $V$ directly as output (continuous), rather than emitting spikes.

---

## 3. Theory → code → behavior (the causal chain)

### 3.1 Autoencoder objective

**Theory:** train parameters $(\theta, \phi)$ so that:

$$z = f_\theta(x), \quad \hat{x} = g_\phi(z), \quad \min \; \mathcal{L}(x, \hat{x}).$$

**In this code:**

- `SpikeAutoEncoder::forward(x)` computes `encoder.forward(x)` then `decoder.forward(z)`.
- The target is set to the input: `criterion->set_target(inputs)`.
- The loss is MSE: `MSELoss` computes $\|\hat{x} - x\|^2$.

**Observed behavior:** if training works, the scalar loss printed every ~10 epochs decreases over time.

### 3.2 Spiking dynamics (LIF)

**Theory (discrete time):** for each neuron and time step,

$$V[t] = \beta V[t-1] + I[t], \quad \beta = \exp(-dt/\tau), \quad \tau = RC.$$

Spiking uses a threshold:

$$S[t] = \Theta(V[t] - V_{th}),$$

followed by a reset (here: hard reset to 0 if spiking).

**In this code:** `LifBPTT::forward` implements exactly this, operating on a flattened input of shape `(Time*Batch, Features)`.

**Observed behavior:** with constant-ish input current, many neurons converge to periodic firing (rate-coded behavior), because the system becomes a stable limit cycle: integrate → spike → reset → repeat.

### 3.3 Rate coding vs. event-based representations

Spikes are **event-based** (sparse in time), but many tasks treat information as **rate-coded**: the value is represented by spike count over a window.

- **Rate-coded view:** a constant input often produces a regular firing rate; reconstruction can rely on average spike rates.
- **Timing-coded view (e.g., TTFS):** precise spike timing carries information; training is typically harder.

**This experiment is closer to rate coding** because the synthetic generator produces a simple, repetitive pattern and the network is trained with MSE on dense time steps.

---

## 4. The model architecture (layer-by-layer)

The demo builds the model in `SpikeAutoEncoder` using two `Sequential` containers.

### 4.1 Encoder

The encoder is:

```
Linear(100 → 50) → LifBPTT
Linear(50  → 40) → LifBPTT
Linear(40  → 30) → LifBPTT
Linear(30  → 20) → LifBPTT
Linear(20  → 10) → LifBPTT
Linear(10  → 10) → LifBPTT
```

The final size `bottleneck_dim` is 10 by default.

### 4.2 Decoder

The decoder mirrors the encoder, ending with a **readout** layer:

```
Linear(10  → 10)  → LifBPTT
Linear(10  → 20)  → LifBPTT
Linear(20  → 30)  → LifBPTT
Linear(30  → 40)  → LifBPTT
Linear(40  → 50)  → LifBPTT
Linear(50  → 100) → LifBPTT(readout_mode=true)
```

### 4.3 Why the final layer is “readout mode”

**Key misconception:** “An autoencoder must output spikes.” Not necessarily.

For reconstruction with MSE, a continuous output is typically easier and better behaved. In snnTorch tutorials, it is common to use the **membrane potential** as the regression output.

**In this code:** the final `LifBPTT` is constructed with `readout_mode=true`, which outputs $V[t]$ directly (no thresholding, no reset). That makes $\hat{x}$ continuous and differentiable at the output.

---

## 5. Time handling and tensor shapes (critical for correctness)

### 5.1 Flattened time-major representation

`LifBPTT` expects the input tensor to have shape:

$$ (T \cdot B, F) $$

where:

- $T$ = `time_steps` (called `steps` in the config)
- $B$ = batch size (inferred)
- $F$ = feature dimension

**Constraint enforced in code:** `input.rows() % time_steps == 0` or `LifBPTT::forward` throws.

### 5.2 How the demo builds `inputs`

The generator returns a sequence of tensors `input_seq[t]` with shape `(B, F)`.
The demo flattens it into a single tensor `(T*B, F)` by writing:

$$\text{inputs}[tB + b, f] = \text{input\_seq}[t][b, f].$$

**Observed behavior:** if you accidentally swap the indexing order (e.g., using `b*T + t`), you will train on scrambled temporal structure and the loss will not behave as expected.

---

## 6. Learning mechanics in detail

### 6.1 Loss function (MSE)

The training objective is:

$$\mathcal{L} = \frac{1}{TBF} \sum_{t,b,f} (\hat{x}[t,b,f] - x[t,b,f])^2.$$

In practice, the demo computes a scalar loss stored in a 1×1 `Tensor`.

### 6.2 Where gradients come from in spiking layers

**The problem:** $S[t] = \Theta(V[t] - V_{th})$ is non-differentiable.

**The solution used here:** the backward pass uses an **exponential surrogate** (configured as `ExponentialSurrogate(1.0f)` when constructing `LifBPTT`). Conceptually:

$$\frac{\partial S}{\partial V} \approx \frac{1}{\alpha} \exp\left(-\frac{|V - V_{th}|}{\alpha}\right).$$

This gives non-zero gradients when $V$ is near threshold. The surrogate does not change the forward spikes, only the learning signal.

### 6.3 Full BPTT in `LifBPTT`

The forward pass stores `v_mem_history` across time when `requires_grad=true`.
The backward pass iterates from $t=T-1$ down to $0$, propagating `grad_next_state` through the decay term $\beta$.

**Causal consequence:** increasing $T$ increases both learning capacity (more temporal context) and memory/computation cost (more history to store).

### 6.4 Parameters that are actually optimized

This is easy to miss:

- Each `LifBPTT` exposes parameters via `params()` as `{&resistance, &voltage_threshold}`.
- Each `Linear` exposes its own weights/bias.
- `SpikeAutoEncoder::params()` concatenates encoder and decoder parameters.

So the optimizer updates both synaptic weights (Linear layers) and neuron parameters (R and threshold) unless you intentionally filter them out.

### 6.5 Gradient clipping

The demo clips gradients by global norm:

1. compute total norm across all parameter gradients,
2. if norm exceeds `max_norm`, scale all gradients.

This prevents exploding gradients, which are common in time-unrolled systems.

---

## 7. Configuration reference (parameters, ranges, constraints)

The demo uses this struct:

```cpp
struct ModelConfig {
    int input_dim;            // F
    int hidden_dims[5];       // fixed-length array
    int bottleneck_dim;
    int steps;                // T
    float dt;                 // seconds
    float R;                  // resistance
    float C;                  // capacitance
    float thr;                // threshold
    float lr;
    int epochs;
    int batch_size;           // declared, not used for batching in this demo
};
```

Practical constraints (to avoid undefined/unstable behavior):

- `steps > 0`.
- `dt > 0`.
- `R > 0`, `C > 0` so that $\tau = RC > 0$ and $\beta = \exp(-dt/\tau)$ is well-defined.
- `thr > 0` (threshold at or below 0 makes spiking degenerate).
- `input.rows() % steps == 0` for any tensor passed into `LifBPTT`.

Typical “sane” ranges (task-dependent, but useful as guardrails):

- $\beta$ close to 1 (slow leak) gives longer memory but can amplify gradient issues; $\beta$ too small forgets too quickly.
- Threshold `thr` too large → almost no spikes (“dead network”); too small → frequent spikes (“spike storm”).

### 7.1 Subtle but important: which neuron parameters are learned

In this framework, each `LifBPTT` layer owns:

- `resistance`: a **1×1 tensor**, i.e., a *scalar* $R$ shared by all units in that layer.
- `voltage_threshold`: a **1×1 tensor**, i.e., a *scalar* $V_{th}$ shared by all units in that layer.

This matches the common “scalar beta/threshold” setup in snnTorch examples. It is not (currently) a per-neuron threshold vector.

---

### 8. Weight loading and initialization (reproducibility)

Runtime NPZ weight loading is disabled in this build. At runtime the demo will not attempt to load `.npz` weight artifacts and will instead initialize weights deterministically using the Kaiming initializer. If you need to produce or consume offline NPZ artifacts for compatibility with other toolchains, use the conversion scripts in `scripts/` (for example `scripts/mat_to_npz.py`) to generate NPZ files outside of the runtime and treat them as offline artifacts.

The demo initializes Linear layers using Kaiming initialization and then scales weights by `0.01` to reduce the risk of immediate saturation in deep SNN stacks.

If you re-enable NPZ loading in a future build, note these format assumptions (for tool-generated NPZ files):

- The loader iterates over `Sequential::layers` and only loads weights for layers that can be `dynamic_pointer_cast<Linear>`.
- Keys are derived from the **layer index in the Sequential container**, not from “linear layer number”. Because the sequence alternates `[Linear, LifBPTT, Linear, LifBPTT, ...]`, weights typically appear at even indices.
- For an index `i`, files are expected to contain `"<i>.weight"` and `"<i>.bias"` entries.

Practical implication: if you change the ordering or insert/remove layers, old `.npz` files will stop matching the architecture. Prefer offline conversion scripts for reproducible artifact generation.

---

## 9. Expected runtime behavior (what “working” looks like)

1. The program prints CPU vectorization support.
2. It loads weights or initializes them.
3. Training begins, printing a loss value periodically.
4. `cpp_loss_log.txt` grows over time.

You should expect:

- loss to generally decrease (not strictly monotonic),
- occasional plateaus if the network settles into a stable firing regime,
- sensitivity to `thr`, `R`, `C`, and the initialization scale.

---

## 10. Common pitfalls and misconceptions (practical debugging guide)

### 10.1 “My loss doesn’t go down — BPTT must be broken”

Often false. In spiking systems, the more common causes are:

- thresholds too high (no spikes → weak learning signal),
- thresholds too low (always spiking → little discrimination),
- weights too large (saturation) or too small (silence).

### 10.2 State handling across batches

Spiking layers are stateful because they keep `v_mem`.
If you do SGD over multiple independent samples, you must reset state between samples/batches.

**In this demo:** `model.reset_states()` is called each epoch before forward.

### 10.3 “batch_size” is not actually used

The config includes `batch_size`, but this particular demo constructs `inputs` with `n_samples = 10` and trains on the whole flattened tensor each epoch. There is no minibatching loop.

**Implication:** optimization behavior differs from a true minibatch SGD setup (gradient noise is lower; learning can be more stable but less representative).

### 10.4 Loss on early time steps (warm-up)

When an SNN starts from $V=0$, the first few steps can be “charging up” transients. Penalizing these equally can slow training.

This demo computes loss on all time steps for simplicity. If you extend this experiment, consider masking early steps.

### 10.5 Confusing “LeakyReLU” with “Lif LIF”

- **Lif ReLU:** a piecewise-linear activation $\max(\alpha x, x)$ in standard ANNs.
- **Lif LIF:** exponential decay of membrane potential in spiking neurons.

This experiment uses the latter.

---

## 11. How to extend this experiment safely

Once the mechanics are validated, typical next steps are:

- Replace the synthetic generator with real data (and define an explicit spike encoding if needed).
- Introduce minibatching (use `DataLoader`) and keep state resets correct.
- Add metrics beyond MSE (e.g., spike rate statistics, sparsity penalties).

If you do any of these, re-check the shape constraint `(T*B, F)` at every spiking layer boundary.

---

## 12. Code map (functions and responsibilities)

This section documents the major functions in the demo in “what it does / why it exists” form.

### 12.1 `SpikeAutoEncoder`

- `SpikeAutoEncoder::SpikeAutoEncoder(cfg)`
    - Builds the encoder and decoder `Sequential` stacks.
    - Uses `LifBPTT(cfg.steps, cfg.dt, cfg.R, cfg.C, cfg.thr, ..., readout_mode)` for spiking/non-spiking behavior.
- `forward(x, requires_grad)`
    - Runs encoder then decoder.
    - `requires_grad=true` is important: it enables `LifBPTT` to store history needed for BPTT.
- `backward(grad_output)`
    - Backpropagates through decoder then encoder.
- `params()`
    - Concatenates encoder and decoder parameters for the optimizer.
- `reset_states()`
    - Calls `nn::utility::reset(...)` on both sequentials to clear `v_mem` in each `LifBPTT`.

### 12.2 Weight init / load

- `initialize_weights(enc_path, dec_path)`
    - Tries to load weights from NPZ; otherwise falls back to Kaiming init + scale.
- `load_weights_from_file(seq, file)`
    - Loads only `Linear` weights/biases; ignores spiking layer parameters.

### 12.3 Optimization helpers

- `clip_gradients(params, max_norm)`
    - Computes the global gradient norm over all params.
    - Scales each gradient tensor in-place if the norm exceeds `max_norm`.
    - Note: `Tensor::grad()` returns a tensor value (copy), so scaling must be followed by `set_grad(...)`.

### 12.4 Main training loop

- Builds synthetic time-series inputs, flattens to `(T*B, F)`.
- Per epoch:
    - `optimizer.zero_grad(params)`
    - `model.reset_states()`
    - forward → loss → backward
    - clip → `optimizer.step(params)`
    - log to `cpp_loss_log.txt`
