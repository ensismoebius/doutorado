# Spike Encoding — Plain Language Guide

> **Technical reference:** [Spike Encoding](../Spike-Encoding.md)

---

## What is spike encoding?

Spiking neural networks (SNNs) communicate using binary spikes — at each time step, a neuron either fires (1) or doesn't (0). But raw input data (speech energy, EEG voltage) is continuous — a float like 0.732.

**Spike encoding** is the conversion from continuous values to spike trains that an SNN can process.

---

## Method 1: Rate coding

The simplest idea: use the *firing rate* to represent magnitude.

- High value → many spikes over the time window
- Low value → few spikes
- Zero → no spikes

Implementation: at each time step, flip a biased coin with probability proportional to the input value. If the coin comes up heads → spike; tails → silence. This is **Poisson rate coding** (Poisson because the spikes are independent random events with a fixed rate).

```
Input = 0.8:    1 1 0 1 1 0 1 0 1 1  (8 spikes in 10 steps)
Input = 0.2:    0 0 1 0 0 0 1 0 0 0  (2 spikes in 10 steps)
Input = 0.0:    0 0 0 0 0 0 0 0 0 0  (0 spikes)
```

**Problem**: To read the rate, you need to count spikes over many time steps. The more steps you use, the more accurate the estimate — but slow. With only 4 steps, the estimate is unreliable (statistical noise).

---

## Method 2: Latency coding (time-to-first-spike)

Instead of counting spikes, encode the value in *when* the first spike fires:

- High value → spike fires early (small time index)
- Low value → spike fires late (large time index)
- Very low value → no spike at all

```
Input = 0.9:  1 0 0 0 0 0 0 0 0 0  (spike at step 1 = high value)
Input = 0.5:  0 0 0 0 1 0 0 0 0 0  (spike at step 5 = medium value)
Input = 0.1:  0 0 0 0 0 0 0 0 0 1  (spike at step 9 = low value)
```

Each neuron fires **at most once** per window. One spike carries one value. This is extremely energy-efficient: 10 neurons with 10 time steps use at most 10 spikes total (vs. potentially 100 for rate coding).

---

## Critical rule: encoding must match loss function

This is one of the easiest mistakes to make. The loss function must match what the encoding treats as information:

| Encoding | Informative quantity | Correct loss |
|---|---|---|
| Rate coding | How many spikes? | `SpikeCountLoss` — compares total spike counts |
| Latency coding | When did the first spike happen? | `SpikeTimeLoss` — compares first-spike times |
| Continuous (classical ANN) | What is the value? | `MSELoss` — compares values directly |

Using the wrong loss means the network tries to optimise the wrong thing. For example, using `SpikeTimeLoss` with rate-coded outputs: the loss only cares about the first spike and ignores all subsequent spikes, making the gradient wrong for all neurons that fire more than once.

---

## Which to use for speech/EEG?

**Rate coding** is simpler and more noise-robust because it averages over many time steps. Good for signals where amplitude is the primary feature.

**Latency coding** is more energy-efficient (fewer spikes) and requires fewer time steps to encode one value. Better for sharp transient signals. EEG signals have important transients, making latency coding potentially more appropriate.

**Derivative coding** (a third option): instead of encoding the signal value, encode changes — fire a spike when the signal rises sharply. This captures events (onset of a word, a sharp EEG transient) rather than sustained amplitude.

---

## The no-spike problem

What happens in latency coding when the input is so low that the neuron never fires within the time window? The first-spike time is undefined.

The implementation handles this by assigning a penalty time equal to the window length T. This is a finite (not infinite) penalty, which keeps the gradient well-defined. But it means the penalty for "never fires" equals the penalty for "fires at the last step" — if this distinction matters for your problem, you may need to tune T.

---

## See also

- [Spike Encoding (technical)](../Spike-Encoding.md) — formulas, SpikeCountLoss/SpikeTimeLoss API
- [SNN and Surrogate Gradients (plain)](./SNN-and-Surrogate-Gradients.md) — the spiking neurons that receive encoded inputs
- [Autoencoders (plain)](./Autoencoders.md) — encoding/loss alignment table
