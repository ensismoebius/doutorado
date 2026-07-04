# Ground-Truth and Smoke Testing

Two testing layers added to catch classes of bugs that unit tests and compilation miss:

1. **PyTorch / snnTorch parity** — numerical ground-truth for individual layers.
2. **Experiment05 smoke tests** — end-to-end runs of every profile to surface runtime errors.

---

## PyTorch / snnTorch parity tests

Compare this C++ library against reference implementations (PyTorch, snnTorch)
element-by-element.

### How it works

- `scripts/testing/gen_pytorch_refs.py` (developer step, needs `torch` + `snntorch`)
  builds each covered layer with fixed seeded weights, runs forward/backward, and
  writes inputs/weights/outputs/gradients as float32 arrays into
  `src/core/layers/tests/fixtures/pytorch_refs.npz` (committed).
- `pytorch_parity_gtest` (C++, no torch at build/CI time) loads that `.npz` with
  cnpy, sets the same weights + input into our layers, runs forward/backward, and
  asserts `EXPECT_NEAR` against the references. Registered in ctest, runs in CI.

The `.npz` is committed (whitelisted in `.gitignore`) so CI needs no torch.

### Coverage

| Layer | Reference | Checked |
|---|---|---|
| Linear | torch.nn.Linear | forward + grad_input + grad_weight + grad_bias |
| Tanh / Sigmoid / ReLU / LeakyReLU | torch | forward + backward |
| MSELoss (mean) | torch | loss + grad_pred |
| CrossEntropyLoss | torch F.cross_entropy | loss + grad_logits (one-hot targets) |
| LSTMLayer | torch.nn.LSTM | forward (batched 3-D) |
| LifBPTT | snnTorch `snn.Leaky` | forward: spike train (subtract + zero reset) + membrane |
| LifBPTT (readout) | snnTorch autograd | backward: exact BPTT temporal gradient dL/dinput |
| Conv1d | torch.nn.Conv1d | forward (valid, stride 1) |
| Conv2d | torch.nn.Conv2d | forward (valid, stride 1, square kernel) |
| MaxPool1d / MaxPool2d | torch | forward |

Exact-math layers match to `1e-4`. Two need care:

- **LSTMLayer** uses rational-approximation activations (`FastActivations.hpp`), so
  it cannot bit-match torch. Checked exactly against a numpy reference using the
  *same* approximation (proves gate order i,f,o,g vs torch's i,f,g,o / recurrence /
  weight orientation), and within a documented bound (`< 0.25`) vs torch's exact LSTM.
- **LifBPTT** maps to `snn.Leaky`: `v[t]=β·v[t-1]+input`, spike on `v>V_th`,
  subtract/zero reset. We set `R=C=1`, `time_step=-ln(β)` so `β=exp(-time_step/(R·C))`
  reproduces snnTorch's β. Spike trains match exactly. The **readout backward** test
  checks `dL/dinput` against snnTorch autograd — exact, since readout mode has no
  spike/reset/surrogate. Spiking-mode backward is *not* parity-tested (surrogate
  gradient + approximate reset-path handling → no exact ground truth).

Convolution weights are stored **pre-permuted into our im2col layout** so the C++
test can `set_weights()`:
`Conv1d[ic*K+k,oc]=torch[oc,ic,k]`, `Conv2d[ic*K*K+ky*K+kx,oc]=torch[oc,ic,ky,kx]`.

### Regenerate / extend

```bash
software/nn/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
software/nn/.venv/bin/python -m pip install snntorch
software/nn/.venv/bin/python software/nn/scripts/testing/gen_pytorch_refs.py
ctest --test-dir out/build/max-performance -R PyTorchParity
```

Add a layer: add a case block in `gen_pytorch_refs.py` (save input/weights/output/
grads), add a `TEST(PyTorchParity, <Layer>)` in `pytorch_parity_gtest.cpp`, regenerate.

See `software/nn/scripts/testing/README.md` for the full contract.

---

## Experiment05 smoke tests

Runtime errors (empty folds, shape mismatches, degenerate data) only appear when a
profile actually **runs**. The smoke suite runs every profile end-to-end with tiny
parameters so those surface quickly.

### Layout

`src/experiments/05/profiles/smoke/` mirrors every real profile
(`debug.json`, `phase00/*`, `phase01/*` → 315 total) keeping every code-path
selector (strategy, wavelet, scale, cepstral, modality, fusion_mode,
classifier.type, nested_cv, standardize_features) but shrinking run parameters:
`repeats=1`, `epochs=2`, `k_folds=2`, `samples_per_batch=4`, `max_samples=60`
(phase00) / `120` (phase01), `results_dir=results/smoke`.

### Automated mirroring (CMake)

The `e05_smoke_profiles` target (in `ALL`) regenerates the mirror whenever any
source profile or `make_smoke_profiles.py` changes — a normal `cmake --build`
keeps `profiles/smoke/` in sync. `GLOB_RECURSE ... CONFIGURE_DEPENDS` picks up
added/removed profiles; the generator is stdlib-only (any `python3`, no torch).

### Running

```bash
cmake --build out/build/max-performance --target experiment05 -j$(nproc)
./scripts/testing/run_smoke.sh [phase00|phase01|all]   # needs the dataset present
```

The runner reports `PASS`/`FAIL` per profile with the captured error line and a
summary. ~315 runs — run on the data machine, without a short timeout.

### Bug it caught

Smoke testing the (previously never-run) DSNN phase01 profiles surfaced
`GroupKFoldPolicy: number of unique groups is less than n_splits`. Root cause:
`max_samples` truncation was `resize(first N)`, but samples are stored
**subject-contiguous** (~130 trials/speaker), so a capped run loaded only 2–3
speakers and nested CV's inner GroupKFold got fewer groups than folds. Fixed in
`E05Dataset.cpp`: truncation now **round-robins across subjects**, so a capped set
spans every speaker (also fixes `debug.json` and any capped run).

---

## Related

- [Test Quality and Determinism](./Test-Quality-and-Determinism.md)
- [Experiment05](../Experiments/Experiment05.md)
- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md)
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md)
