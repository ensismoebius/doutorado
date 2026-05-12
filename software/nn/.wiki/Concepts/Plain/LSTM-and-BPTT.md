# LSTM and BPTT — Plain Language Guide

> **Technical reference:** [LSTM and BPTT](../LSTM-and-BPTT.md)

---

## The problem: neural networks that forget too quickly

A standard neural network processes each input independently. Feed it one speech frame, get an answer. Feed it the next frame, it has no memory of what came before.

For sequential signals like speech or EEG, this is a problem. The network needs to "remember" earlier parts of the signal to understand what's happening now.

The first solution was the **Recurrent Neural Network (RNN)**. At each step it passes a "hidden state" forward — a summary of everything seen so far. But in practice, RNNs fail badly on long sequences because the gradient (the learning signal) shrinks exponentially as it travels backwards through time. After ~10–20 steps, the gradient is essentially zero — the network cannot learn long-range patterns.

---

## LSTM: a memory cell with gatekeepers

The **Long Short-Term Memory (LSTM)** network solves this by replacing the simple hidden state with a **cell state** — a separate memory that passes through time with very little modification, like a conveyor belt.

The key innovation: four *gates* control what gets written to, read from, and erased from the cell.

### The four gates, in plain terms

| Gate | Job | Analogy |
|---|---|---|
| **Forget gate** | What old memory to throw away | "Is this information still relevant?" |
| **Input gate** | How much new information to write | "Is this new input worth remembering?" |
| **Cell candidate** | What new information to potentially write | "What exactly should I write down?" |
| **Output gate** | What to read out from the cell now | "What from memory is relevant right now?" |

All four gates output values between 0 and 1 (using sigmoid). A gate value of 0 means "pass nothing"; 1 means "pass everything".

### Why the cell state prevents vanishing gradients

The cell update is: `new_cell = (forget × old_cell) + (input × new_candidate)`

Notice: the recurrent path `forget × old_cell` has **no weight matrix** on it. In a standard RNN, the recurrent path is `W × hidden`, and repeatedly multiplying by W makes the gradient vanish. Here, the multiplication is by the forget gate value — a number between 0 and 1, but crucially it is *learned* and can stay close to 1 for sequences where memory needs to persist.

---

## BPTT: how LSTMs learn from sequences

**Backpropagation Through Time (BPTT)** is just regular backpropagation applied to an unrolled network. Imagine you have a 10-step sequence:

```
step 1 → step 2 → step 3 → ... → step 10 → loss
```

Backpropagation computes the gradient by going backwards:

```
loss → step 10 → step 9 → ... → step 1
```

At each step, the gradient gets passed back along both the output connection *and* the recurrent cell state connection. LSTM's cell state gradient path has this "+1" property — even if everything else shrinks, the gradient can travel back along the cell state path without vanishing.

---

## Forget gate bias initialised to 1

A small but important detail: the forget gate is initialised to be "mostly open" (bias = 1). This means at the start of training, the LSTM tends to *keep* old memory rather than forget it. This makes early training more stable.

---

## LSTM vs SNN in this project

The project compares two architectures for feature extraction:

| Architecture | Recurrence | Memory mechanism | Gradient |
|---|---|---|---|
| LSTM-AE | Yes, via hidden/cell state | Forget/input/output gates | Exact BPTT |
| SNN-AE | Yes, via membrane potential | Threshold/reset dynamics | Approximate (surrogate) |

Both process sequences. LSTM has exact, well-understood training. SNN has biologically-inspired dynamics and is more energy-efficient.

---

## In plain terms: what does the LSTM actually do here?

1. Takes a window of speech or EEG data as a sequence (e.g., 64 values per timestep, 32 timesteps)
2. Processes it step by step, maintaining a "running summary" (hidden + cell state)
3. After all steps, the final hidden state is the compressed representation of the whole sequence
4. This compressed vector is the input to the classifier

---

## See also

- [LSTM and BPTT (technical)](../LSTM-and-BPTT.md) — gate equations, BPTT formulas, implementation
- [SNN and Surrogate Gradients (plain)](./SNN-and-Surrogate-Gradients.md) — the spiking alternative
- [Autoencoders (plain)](./Autoencoders.md) — the LSTM is used inside an autoencoder
