# testing/

This directory holds two unrelated things, both under "testing": ground-truth
parity fixtures (this README) and the Experiment05 profile runners. See the
top-level `scripts/README.md` for the full file table; summary:

| Script | Role |
|---|---|
| `gen_pytorch_refs.py` | PyTorch parity fixtures (documented below) |
| `gen_pywt_refs.py` | PyWavelets ground-truth fixtures for the C++ wavelet ops (needs `pywt`); see header comment for the periodization/DaubN conventions mapped from `waveletOperations.cpp` |
| `e05_make_smoke_profiles.py` + `run_e05_smoke.sh` | Fast smoke pass over every Experiment05 profile (tiny params, same code paths) |
| `run_e05_profiles.sh` | The real (heavy) Experiment05 phase00/phase01 runner — resumable, checkpointed |
| `test_e05_phase_scripts.py` | Unit tests for `../pipeline/e05_phase00_rank.py` + `../pipeline/e05_apply_winner.py` |

---

# PyTorch parity tests

Ground-truth tests that compare this C++ library against PyTorch.

## How it works

- `gen_pytorch_refs.py` (developer step, needs `torch`) builds each covered layer
  in PyTorch with fixed seeded weights, runs forward/backward, and writes the
  inputs/weights/outputs/gradients as float32 arrays into
  `../../src/core/tensor/tests/fixtures/pytorch_refs.npz` (committed).
- `pytorch_parity_gtest` (C++, no torch needed) loads that `.npz` with cnpy, sets
  the same weights + input into our layers, runs forward/backward, and asserts
  `EXPECT_NEAR` against the PyTorch references. Runs in CI via ctest.

The `.npz` is committed (whitelisted in `.gitignore`) so CI needs no torch.

Every test is a `TYPED_TEST` run once per concrete tensor backend
(`XTensorBackend`, `OpenCLTensorBackend`, `DeviceTensorBackend`, and
`SYCLTensorBackend` when `NN_BACKEND=SYCL`) against the same fixture — not
just whichever backend the current build selected as `nn::Backend`. Lives
under `src/core/tensor/tests/` (not `src/core/layers/tests/`) because naming
concrete backend types is restricted to that zone by
`cmake/BackendImplementationGuard.cmake`. See
`.wiki/Guides/Ground-Truth-and-Smoke-Testing.md` for the two real bugs this
cross-backend run caught (a hardcoded-`nn::Tensor` bug in
`FastActivations.hpp` and missing contract methods on `DeviceTensorBackend`).

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
2. Add a `TYPED_TEST(PyTorchParityTyped, <Layer>)` in
   `../../src/core/tensor/tests/pytorch_parity_gtest.cpp` using the layer's
   `Impl<B>` template (e.g. `LinearImpl<B>`, not the `nn::Linear` alias, which
   is tied to whichever single backend is currently selected) that loads those
   keys, sets the weights, runs the op, and compares.
3. Regenerate the fixtures and rebuild.

**Backend note (important).** These tests run on all four backends: xtensor
(row-major), OpenCL (column-major), Device (row-major, host mirror), and SYCL
(row-major, host mirror + optional device dispatch). Use only the *structured*
accessors `at(i,j)` /
`at(i,j,k)` / `at(i,j,k,l)` — never the linear `at(k)`, which exposes backend
storage order and would transpose a tensor filled from row-major fixture data on
OpenCL. The helpers (`make_from`, `fill_from`, `expect_close`) already enforce
this; `expect_close` walks the tensor's own logical shape and reads the C-order
fixture in lockstep, so it also tolerates a rank difference that flattens the same
way (e.g. our 2-D `(T*B,F)` LIF output vs the 3-D `(T,B,F)` reference). New two-plus
dimensional tensors must be filled/compared through these helpers, not by hand.
