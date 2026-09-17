# Tensor — Plain Language Guide

> **Technical reference:** [Tensor](../Tensor.md)

---

## What is a tensor?

A **scalar** is a single number: `3.14`  
A **vector** is a list of numbers: `[1.0, 2.5, 0.7, 3.1]`  
A **matrix** is a 2D grid of numbers (rows and columns)  
A **tensor** is the generalisation: it can be 1D, 2D, 3D, or any number of dimensions

In neural networks, almost everything is a tensor:
- A single training sample (e.g., 13 LFCC coefficients) → 1D tensor, shape `(13,)`
- A batch of 32 samples → 2D tensor, shape `(32, 13)`
- A batch of sequences of length 10 → 3D tensor, shape `(32, 10, 13)`

Tensors are to neural network code what arrays are to regular programming — the fundamental data type.

---

## Why not just use C++ arrays?

The `Tensor` class in this library provides extra capabilities beyond a plain array:

1. **Automatic gradient tracking**: when you ask the tensor to track gradients (`requires_grad=true`), every operation on it records what happened so that backpropagation can work out how to update the weights.

2. **Backend dispatch**: the same code works against a swappable backend — by default the `xtensor` CPU backend. The calling code doesn't need to know which backend is compiled in.

3. **Named operations**: `matmul`, `element_wise_add`, `reshape` etc. are readable and checked for shape compatibility.

---

## Gradient tracking in plain terms

During the **forward pass**, you compute: `output = weights * input + bias`

During the **backward pass**, you need to know: "how much does the loss change if I change each weight by a tiny amount?"

If you just used raw arrays, you'd have to manually derive and code every gradient. Gradient tracking means the tensor remembers the computation path: "I was created by multiplying these two tensors." Backpropagation then walks that recorded path backwards, applying the chain rule automatically.

Calling `forward(input, requires_grad=true)` tells the layer to record this path. Calling `backward(gradient)` walks it back.

---

## Shape conventions in this library

| Use case | Shape | Meaning |
|---|---|---|
| Feature vectors | `(batch_size, features)` | rows = samples, cols = features |
| Sequences for LSTM | `(batch, time_steps, features)` | 3D |
| SNN time-major | `(T * batch, features)` | all time steps of all samples stacked |

The SNN convention is unusual: instead of `(batch, time, features)`, all time steps are packed flat: `(T*B, F)`. The first `B` rows are time step 0 for all samples, the next `B` rows are time step 1, etc. This flat layout is faster for the spiking neuron implementation.

---

## See also

- [Tensor (technical)](../Tensor.md) — API reference, backend details
- [Layers (plain)](../Layers.md) — how tensors flow through layers
- [Training (plain)](../Training.md) — what happens to tensors during training
