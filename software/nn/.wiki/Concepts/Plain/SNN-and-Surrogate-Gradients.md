# Spiking Neural Networks and Surrogate Gradients — Plain Language Guide

> **Technical reference:** [SNN and Surrogate Gradients](../SNN-and-Surrogate-Gradients.md)

---

## What is a spiking neural network?

Regular ("classical") neural networks pass numbers between neurons at every step. A neuron multiplies its inputs by weights, adds them up, and passes the result to the next layer continuously.

A **spiking neural network (SNN)** is different: neurons communicate using brief electrical pulses called **spikes** (or action potentials) — the same mechanism real biological neurons use. A spike is binary: either it fires (1) or it doesn't (0). Information is carried in the *pattern and timing* of spikes, not in continuous values.

---

## How does a spiking neuron work?

The specific neuron model used here is the **Leaky Integrate-and-Fire (LIF)** neuron. Think of it like a leaky bucket:

```
Water pours in (input current)
Bucket fills up (membrane voltage rises)
Water slowly leaks out (membrane voltage decays)
When bucket overflows (voltage > threshold) → SPIKE!
Bucket is immediately emptied (voltage reset to 0)
```

More precisely:
- **Integrate**: The neuron accumulates incoming signals. Each input adds to the membrane voltage.
- **Leak**: Between inputs, the voltage slowly decays back toward zero (with decay rate β = exp(−Δt/RC)).
- **Fire**: When voltage exceeds a threshold, the neuron fires (emits a 1) and the voltage resets.

The trainable parameters R (resistance) and C (capacitance) control how quickly the neuron forgets previous inputs. A large RC time constant = long memory; small RC = forgets quickly.

---

## Why are SNNs interesting?

1. **Energy efficiency**: A spiking neuron only "does work" when it fires. If neurons fire at 10% of time steps (sparse activity), the network uses roughly 10× less energy than an equivalent classical network. This matters for edge devices and brain-computer interfaces.

2. **Temporal dynamics**: SNNs naturally represent time — different firing patterns across time steps encode different information. This fits well with time-varying signals like EEG and speech.

3. **Biological plausibility**: Real brains use spikes. SNN dynamics are closer to what actually happens in cortical neurons.

---

## The training problem: spikes are not differentiable

Training neural networks requires computing gradients — how much does the loss change when you change a weight? For classical networks, this uses the chain rule of calculus.

But the spike function is a step: the neuron either fires or doesn't. A step function has a gradient of **zero almost everywhere** (and is undefined at the threshold). Zero gradient → no learning signal → the network cannot be trained.

This is called the **non-differentiability problem**.

---

## The solution: surrogate gradients

During the **forward pass** (computing the output), the network uses the real spike rule: fire if voltage > threshold.

During the **backward pass** (computing gradients), instead of using the true gradient (which is zero), the network pretends the spike function has a smooth, differentiable shape — the **surrogate gradient**.

The project uses the **exponential surrogate** (also called SuperSpike):

```
True gradient: 0 everywhere except exactly at threshold (undefined)
Surrogate:     β × exp(−β × |voltage − threshold|)
              ↑ a smooth bump peaked at the threshold
```

This is a "white lie" told only during backpropagation. It gives a non-zero gradient for neurons close to threshold, enabling the weights to be adjusted in the right direction.

---

## Spike-frequency adaptation

A problem with simple LIF neurons: once a neuron finds a good threshold, it may fire on every single input ("bursting") or never fire at all ("dead neurons"). 

Adaptation adds a dynamic threshold: after each spike, the threshold rises temporarily, making it harder to fire again immediately. After some time without spiking, the threshold falls back to its baseline. This creates more selective, informative firing patterns.

---

## Threshold-Dependent Batch Normalization (tdBN)

In deep SNNs (many layers), the voltages can spiral out of control — either growing too large (explosions) or shrinking to zero. tdBN normalises the voltage before each layer, similar to regular batch normalisation in classical networks, but calibrated to the spiking threshold. This stabilises training of deeper networks.

---

## BPTT for SNNs

When the SNN processes a sequence of time steps, gradients are computed by **Backpropagation Through Time (BPTT)** — unrolling the time steps and backpropagating through all of them at once. The surrogate gradient is applied at each time step where spikes occur.

---

## Where SNNs are used in this project

The SNN is used as an **autoencoder**: it compresses speech/EEG signals into a compact representation, then reconstructs them. The latent representation (compressed output) is used as features for speaker verification.

The comparison is: does a spiking autoencoder learn better features than an LSTM autoencoder? Results are compared using the paraconsistent quality metric (D_truth distance).

---

## See also

- [SNN and Surrogate Gradients (technical)](../SNN-and-Surrogate-Gradients.md) — LIF equations, surrogate formulas, all layer types
- [Spike Encoding (plain)](./Spike-Encoding.md) — how continuous signals become spikes
- [Spike Rate Regularization (plain)](./Spike-Rate-Regularization.md) — preventing dead/bursting neurons
- [Autoencoders (plain)](./Autoencoders.md) — the autoencoder context
- [LSTM and BPTT (plain)](./LSTM-and-BPTT.md) — the classical comparison
