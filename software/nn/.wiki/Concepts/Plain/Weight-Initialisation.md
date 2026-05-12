# Weight Initialisation — Plain Language Guide

> **Technical reference:** [Weight Initialisation](../Weight-Initialisation.md)

---

## Why initialisation matters

Before training begins, all the weights in a neural network need to start with some initial values. The choice of those starting values has a surprisingly large effect on whether training works at all.

Think of it like orienteering: if you start at the right location, you can find the goal quickly. Start in a swamp, and you may never escape.

---

## The two failure modes

### All zeros (the obvious wrong answer)

If all weights start at zero, every neuron computes the same thing. All gradients are also identical. After the first update, all neurons still have identical weights. No matter how long you train, the network cannot break this symmetry — it effectively has one neuron, not thousands.

### All the same value (same problem)

The symmetry problem applies to any constant, not just zero. Each layer must start with varied weights.

### Too large values

If weights start large, signals grow exponentially as they pass through layers. Activations saturate (sigmoid outputs all ~1, tanh outputs all ~1 or -1), and gradients vanish. The network learns nothing.

### Too small values

If weights start tiny, signals shrink exponentially. After 10 layers, everything is near zero. Gradients also vanish. Same result: no learning.

---

## The goal: keep signal magnitude stable across layers

The ideal initialisation keeps the typical size of activations (and gradients) roughly constant from input to output, regardless of how many layers there are.

The mathematically correct scale depends on:
- How many inputs each neuron has (fan-in)
- How many outputs each neuron sends to (fan-out)
- What activation function is used after the layer

---

## Xavier / Glorot initialisation

Designed for **sigmoid** and **tanh** activations. It calculates the appropriate weight scale from both fan-in and fan-out:

```
scale = sqrt(6 / (fan_in + fan_out))   ← for uniform distribution
scale = sqrt(2 / (fan_in + fan_out))   ← for Gaussian distribution
```

Weights are drawn randomly from a distribution with this scale. The derivation ensures that the variance of activations stays roughly constant going forward, and the variance of gradients stays roughly constant going backward.

---

## Kaiming / He initialisation

Designed for **ReLU** activations. ReLU throws away exactly half of its inputs (everything negative becomes zero). So the "effective" fan-in is half the actual fan-in. Kaiming accounts for this:

```
scale = sqrt(2 / fan_in)
```

The factor of 2 compensates for the halving effect of ReLU.

For spiking neurons (LIF/Lif), which also have a non-linear threshold and reset, the formula is adapted to account for the leak rate.

---

## Practical rules

| Activation | Recommended initialisation |
|---|---|
| Sigmoid, Tanh | Xavier (Glorot) |
| ReLU, LeakyReLU | Kaiming (He) |
| LIF spiking neurons | Kaiming adapted for leak rate |
| None (linear output) | Xavier |

**Bias initialisation**: always zero. The bias shifts the activation function; starting at zero is almost always correct. Exception: LSTM's forget gate bias should start at 1 (to initially keep memory — explained in the LSTM page).

---

## Why this matters for SNNs specifically

Spiking neurons are sensitive to initial weight scale. Too small → no neuron ever reaches threshold → all neurons are immediately dead (zero output, zero gradient). Too large → all neurons fire constantly → bursting from step one. The adapted Kaiming initialisation tries to set weights such that neurons start near their threshold with moderate firing rates.

---

## See also

- [Weight Initialisation (technical)](../Weight-Initialisation.md) — formulas and implementation
- [SNN and Surrogate Gradients (plain)](./SNN-and-Surrogate-Gradients.md) — why SNN initialisation is tricky
- [Residual Blocks (plain)](./Residual-Blocks.md) — why deep networks need good initialisation
- [Adam Optimiser (plain)](./Adam-Optimiser.md) — what runs after initialisation
