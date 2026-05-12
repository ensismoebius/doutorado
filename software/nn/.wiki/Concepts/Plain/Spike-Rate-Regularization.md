# Spike Rate Regularization — Plain Language Guide

> **Technical reference:** [Spike Rate Regularization](../Spike-Rate-Regularization.md)

---

## Two ways SNN training goes wrong

Without special care, spiking autoencoders tend to fall into one of two broken states:

### Dead neurons
A neuron's voltage never reaches threshold, so it never fires. Its output is always zero. Backpropagation through the surrogate gradient gives near-zero gradients for neurons far from threshold — so the weights stop updating and the neuron is permanently stuck silent.

This is analogous to the "dying ReLU" problem in classical networks (neurons always outputting zero).

### Bursting neurons
The opposite extreme: a neuron fires at nearly every time step. Its output is almost always 1. This neuron has stopped being informative — it's just a constant bias. The gradients become noisy and the neuron's weights stop improving.

In both cases, the neuron contributes nothing useful to the network's task.

---

## The solution: penalty for being out of range

The fix is to add a soft penalty to the training loss that pushes the network-wide average firing rate toward a healthy target range — say, 5% to 30% of time steps.

In plain terms:
- **Below 5%?** Add a penalty proportional to how far below 5% you are. The gradient pushes neurons to fire more.
- **Above 30%?** Add a penalty proportional to how far above 30% you are. The gradient pushes neurons to fire less.
- **Between 5% and 30%?** No penalty. Let the reconstruction loss guide the network.

The total training loss becomes: `reconstruction_loss + λ × rate_penalty`

The weight λ controls how strongly the regularisation enforces the firing rate target. A larger λ ensures healthy firing rates but may hurt reconstruction quality. Typical starting values: λ = 0.001–0.01.

---

## Why this range?

Research shows that networks with 10–30% mean firing rate achieve the best reconstruction while maintaining sparsity (and therefore energy efficiency). Below ~5%, neurons are effectively dead. Above ~80%, neurons are bursting.

The bounds in this project default to: `min_rate = 0.05` (5%), `max_rate = 0.80` (80%).

---

## Tracking spike rates

During training, the mean firing rate is logged per epoch in `EpochResult.mean_spike_rate`. This lets you monitor whether the regularization is working:
- Steadily near 0 → regularization is too weak or threshold is too high
- Steadily near 1 → regularization is too weak or threshold is too low
- Healthy oscillation around 10–20% → training is progressing well

---

## Synaptic operations (energy estimate)

One of the reasons to use SNNs is energy efficiency. "Synaptic operation" (SOP) counts how many times a spike travels across a connection. The total energy cost of a forward pass is roughly proportional to the total SOP count:

```
SOPs = (total spikes across all neurons) × (output connections per neuron)
```

For comparison, a classical ANN with the same architecture performs a multiply-accumulate for every connection on every forward pass — equivalent to 100% firing rate. An SNN with 10% firing rate uses approximately 10× fewer operations.

The `EpochResult.sops` field tracks this per epoch.

---

## Combined with adaptation

Spike-frequency adaptation (available in `LifImpl` via `adapt_coupling`) is a complementary mechanism: after each spike, the threshold rises temporarily, preventing immediate re-firing. This naturally suppresses burst mode.

Using both adaptation and rate regularization together gives the best stability — adaptation handles individual neurons while regularization handles the network-wide average.

---

## See also

- [Spike Rate Regularization (technical)](../Spike-Rate-Regularization.md) — regularization formula, gradient, API
- [SNN and Surrogate Gradients (plain)](./SNN-and-Surrogate-Gradients.md) — the neurons being regularized
- [Spike Encoding (plain)](./Spike-Encoding.md) — what spike rates mean
- [Autoencoders (plain)](./Autoencoders.md) — the training context
