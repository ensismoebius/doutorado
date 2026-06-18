# Membrane Dynamics — Plain Language

## The coffee-cup analogy

Imagine a coffee cup with a small hole in the bottom. You pour coffee in (input current), the level rises (membrane potential increases). The hole leaks coffee out over time (the "leaky" part). If the coffee reaches the rim (threshold), the cup overflows — that's a spike — and then the cup is immediately emptied (reset).

This is exactly what a Leaky Integrate-and-Fire (LIF) neuron does, but with electricity instead of coffee.

## The three numbers that control it

| Parameter | What it controls | Default |
|---|---|---|
| **R** (resistance) | How slowly current builds up potential | 1.0 |
| **C** (capacitance) | How much charge the "cup" holds | 1.0 |
| **V_th** (threshold) | The rim height that triggers a spike | 1.0 |

The time constant τ = R × C controls how quickly the "cup" leaks. Large τ = slow leak = neuron integrates over longer time. Small τ = fast leak = neuron responds only to recent input.

## What happens each time step

1. **Leak**: multiply current membrane potential by β (a number < 1, so it shrinks)
2. **Integrate**: add the new input current
3. **Check**: is the potential above threshold?
   - **Yes** → emit spike (output = 1), reset potential to 0
   - **No** → do nothing (output = 0), keep potential for next step

β = exp(−Δt / (R × C)). If R=1, C=1, Δt=1, then β ≈ 0.37 (potential decays to 37% each step).

## Why R and C are trainable

Just like a neural network learns the best weights, this network learns the best "leakiness" for each layer. A layer processing fast events (high-frequency audio) will learn a small τ. A layer integrating over slower rhythms (EEG bands) will learn a large τ.

## The safety clamp

During training the optimiser might accidentally push R or C below zero. That would make τ = R×C negative, and exp(−Δt/τ) would explode. The code always clamps R ≥ 1e-6 and C ≥ 1e-6 as a safety guard.

## Common mistakes

1. **Forgetting to reset between clips**: the neuron remembers its voltage. Call `reset_state()` before each new audio segment.
2. **Learning rate too large**: if Adam takes huge steps, R and C can hit the safety clamp, making them stuck at 1e-6 and untrainable.
3. **Threshold too high**: if V_th is set too large, the neuron never fires and no gradients flow (dead neuron problem).

## See Also

- [Membrane Dynamics — Technical](../Membrane-Dynamics.md)
- [SNN and Surrogate Gradients — Plain](./SNN-and-Surrogate-Gradients.md)
- [Time-Major Layout — Plain](./Time-Major-Layout.md)
