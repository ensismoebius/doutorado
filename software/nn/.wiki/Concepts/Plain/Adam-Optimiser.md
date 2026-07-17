# Adam Optimiser — Plain Language Guide

> **Technical reference:** [Adam Optimiser](../Adam-Optimiser.md)

---

## What is an optimiser?

A neural network learns by adjusting thousands of numbers (called *weights*) until its predictions get better. An optimiser is the algorithm that decides *how much* to change each weight after every batch of training examples.

Think of training as trying to find the lowest point in a hilly landscape while blindfolded. The optimiser is the strategy you use to walk downhill.

---

## The simple approach: follow the slope

The most basic strategy — **gradient descent** — looks at the slope of the loss at your current position and takes a step in the downhill direction. But it uses the same step size for every weight, no matter how steep or flat the terrain is.

Problem: if you use a large step size you might overshoot; if you use a small one you learn very slowly.

---

## What Adam does differently

Adam keeps **memory** about recent steps and uses it to set a different step size for each weight automatically.

It tracks two things for every weight:

1. **Momentum** — a running average of recent gradients (which direction have I mostly been moving?).  
   Like a ball rolling downhill that keeps some of its previous velocity.

2. **Variance** — a running average of how *large* recent gradients have been (is this terrain steep or flat?).  
   Weights that have been getting big gradients get *smaller* steps; weights that have been getting tiny gradients get *larger* steps.

The update rule is roughly:

```
new_weight = old_weight − (learning_rate × momentum) / sqrt(variance)
```

The division by `sqrt(variance)` is what makes it adaptive: noisy, high-variance parameters are stepped cautiously; stable, low-variance ones are stepped more aggressively.

---

## The "cold start" fix

At the very beginning of training, both momentum and variance are 0 (no history yet). This would make the first few updates far too small. Adam corrects for this by dividing by `(1 − decay^step)`, which gets close to 1 quickly after the first few steps. This is called **bias correction**.

---

## Typical settings

| What | Value | Meaning |
|---|---|---|
| Learning rate | 0.001 | Overall step size |
| β₁ | 0.9 | Momentum memory — forgets 10% of old momentum each step |
| β₂ | 0.999 | Variance memory — forgets 0.1% each step (very slow) |
| ε | 1e-8 | Tiny number to prevent dividing by zero |

These defaults work well for most problems and rarely need changing.

---

## Why Adam is used in this project

Training spiking neural networks (SNNs) involves a mix of parameters with very different scales — regular weights, membrane resistance, capacitance, and voltage threshold. Adam's per-parameter step sizes handle this heterogeneity naturally. The SNN biophysical parameters (R, C, V_th) additionally use a 10× smaller learning rate multiplier because their gradients operate on a different scale.

---

## Common mistakes

- **Forgetting to call `zero_grad()` before each batch** — gradients accumulate, making the update incorrect.
- **Using the same learning rate for all parameter types** — SNN physical parameters need a lower rate; use `Optimizer::attach_with_scales()`.
- **Changing ε without a good reason** — only do this if you see NaN losses.

---

## See also

- [Adam Optimiser (technical)](../Adam-Optimiser.md) — full math and implementation
- [Weight Initialisation (plain)](./Weight-Initialisation.md) — what happens before training starts
- [SNN and Surrogate Gradients (plain)](./SNN-and-Surrogate-Gradients.md) — what Adam is optimising
