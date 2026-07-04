# PyTorch parity tests

Ground-truth tests that compare this C++ library against PyTorch.

## How it works

- `gen_pytorch_refs.py` (developer step, needs `torch`) builds each covered layer
  in PyTorch with fixed seeded weights, runs forward/backward, and writes the
  inputs/weights/outputs/gradients as float32 arrays into
  `../../src/core/layers/tests/fixtures/pytorch_refs.npz` (committed).
- `pytorch_parity_gtest` (C++, no torch needed) loads that `.npz` with cnpy, sets
  the same weights + input into our layers, runs forward/backward, and asserts
  `EXPECT_NEAR` against the PyTorch references. Runs in CI via ctest.

The `.npz` is committed (whitelisted in `.gitignore`) so CI needs no torch.

## Coverage

| Layer | Reference | Checked |
|---|---|---|
| Linear | torch.nn.Linear | forward + grad_input + grad_weight + grad_bias |
| Tanh / Sigmoid / ReLU / LeakyReLU | torch | forward + backward |
| MSELoss (mean) | torch | loss + grad_pred |
| LSTMLayer | torch.nn.LSTM | forward (batched 3-D) |
| LifBPTT | snnTorch `snn.Leaky` | forward: spike train (subtract + zero reset) + membrane |
| LifBPTT (readout) | snnTorch autograd | backward: exact BPTT temporal gradient dL/dinput |
| Conv1d | torch.nn.Conv1d | forward (valid, stride 1) |
| Conv2d | torch.nn.Conv2d | forward (valid, stride 1, square kernel) |
| CrossEntropyLoss | torch F.cross_entropy | loss + grad_logits (one-hot targets) |
| MaxPool1d | torch F.max_pool1d | forward |
| MaxPool2d | torch F.max_pool2d | forward |

Exact-math layers match PyTorch to `1e-4`. Two layers need care:

- **LSTMLayer** uses rational-approximation activations (`FastActivations.hpp`) for
  speed, so it can't bit-match torch. Checked exactly against a numpy reference
  using the *same* approximation (proves gate order / recurrence / weight
  orientation), and within a documented bound (`< 0.25`) vs torch's exact LSTM.
- **LifBPTT** maps to snnTorch's `snn.Leaky`: `v[t]=β·v[t-1]+input`, spike on
  `v>V_th`, subtract/zero reset. We set `R=C=1`, `time_step=-ln(β)` so
  `β=exp(-time_step/(R·C))` reproduces snnTorch's `β`. Spike trains match exactly;
  the readout case checks the leaky-integrator membrane. The **readout backward**
  test checks `dL/dinput` (BPTT through the recurrence) against snnTorch autograd —
  exact, since readout mode has no spike/reset/surrogate. Spiking-mode backward is
  *not* parity-tested: it uses a surrogate gradient (our `ExponentialSurrogate` vs
  snnTorch's) plus approximate reset-path handling, so no exact ground truth.

Convolution weights are stored **pre-permuted into our im2col layout** so the C++
test sets them via `set_weights()`:
`Conv1d[ic*K+k, oc]=torch[oc,ic,k]`, `Conv2d[ic*K*K+ky*K+kx, oc]=torch[oc,ic,ky,kx]`.

## Regenerate (after changing a covered layer)

```bash
# one-time: create the venv and install CPU torch + snnTorch
software/nn/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
software/nn/.venv/bin/python -m pip install snntorch

software/nn/.venv/bin/python software/nn/scripts/testing/gen_pytorch_refs.py
```

Then rebuild + run: `ctest --test-dir out/build/max-performance -R PyTorchParity`.

## Add a new layer

1. Add a case block in `gen_pytorch_refs.py` (save input/weights/output/grads).
2. Add a `TEST(PyTorchParity, <Layer>)` in
   `../../src/core/layers/tests/pytorch_parity_gtest.cpp` that loads those keys,
   sets the weights, runs the op, and compares.
3. Regenerate the fixtures and rebuild.
