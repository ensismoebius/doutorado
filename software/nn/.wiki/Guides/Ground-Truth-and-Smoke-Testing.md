# Ground-Truth and Smoke Testing

Testing layers added to catch classes of bugs that unit tests and compilation miss:

1. **PyTorch / snnTorch parity** — numerical ground-truth for individual layers.
1b. **Micro-network parity** — ground-truth for whole small NETWORKS (ANN / SNN / LSTM),
   because every layer can be individually correct while the network built from them is
   wrong. `scripts/testing/gen_micro_network_refs.py` → committed
   `src/core/layers/tests/fixtures/micro_network_refs.npz` → `micro_network_parity_gtest`.
   It found three real defects on its first run: MSELoss/MAELoss silently clipping their own
   gradient at norm 1.0 (Trainer's default loss — it hit every trained autoencoder, and
   overrode `grad_clip_norm=0`); a false claim that our LIF is "exactly snnTorch's snn.Leaky"
   (true only for reset="zero", which is what production uses — subtract diverges ~2-3%, and
   the per-layer fixture missed it by spiking only 3/36 times); and that our LSTM uses
   softsign gates, not sigmoid/tanh (|tanh − rat_tanh| reaches 0.306), so it can never match
   torch.nn.LSTM. See fixme.md for the full write-up.
2. **PyWavelets parity** — ground-truth for the wavelet transforms (see
   [Wavelet](../Core/Wavelet.md#ground-truth-vs-pywavelets-2026-07-15)); it
   caught a `malat` DWT buffer-corruption bug and three corrupted Daubechies
   filter tables.
3. **Experiment05 smoke tests** — end-to-end runs of every profile to surface runtime errors.

Each compares against a mature reference library, stores committed `.npz`
fixtures (so CI needs no Python), and reloads them with cnpy on the C++ side.

---

## PyTorch / snnTorch parity tests

Compare this C++ library against reference implementations (PyTorch, snnTorch)
element-by-element.

### How it works

- `scripts/testing/gen_pytorch_refs.py` (developer step, needs `torch` + `snntorch`)
  builds each covered layer with fixed seeded weights, runs forward/backward, and
  writes inputs/weights/outputs/gradients as float32 arrays into
  `src/core/tensor/tests/fixtures/pytorch_refs.npz` (committed).
- `pytorch_parity_gtest` (C++, no torch at build/CI time) loads that `.npz` with
  cnpy, sets the same weights + input into our layers, runs forward/backward, and
  asserts `EXPECT_NEAR` against the references. Registered in ctest, runs in CI.

The `.npz` is committed (whitelisted in `.gitignore`) so CI needs no torch.

**Every concrete backend, not just whichever one is `nn::Backend`.** Every test
is a `TYPED_TEST` run once per `XTensorBackend`, `OpenCLTensorBackend`,
`DeviceTensorBackend`, and `SYCLTensorBackend` (the last only when
`NN_BACKEND=SYCL`, since its kernels TU is only compiled then) — each backend's
`LinearImpl<X>`/`Conv1dImpl<X>`/etc. is checked against the *same* PyTorch/
snnTorch fixture, independent of which single backend the rest of the project
happens to have selected in that build. OpenCL/SYCL skip gracefully
(`GTEST_SKIP`) when no device is present at run time rather than failing;
XTensor and Device never skip (host math only). Lives in
`src/core/tensor/tests/` rather than `src/core/layers/tests/` because
naming four concrete backend types side-by-side is exactly what
`cmake/BackendImplementationGuard.cmake` restricts to `include/tensor/` /
`src/core/tensor/` — see that file's comments for the "layers stay
backend-agnostic" rule this test is deliberately exempt from.

This cross-backend run caught two real, previously-latent bugs no prior test
exercised (nothing had ever instantiated these templates for a non-default
backend before):
- `FastActivations.hpp`'s `sigmoid_fast_tensor`/`tanh_fast_tensor`/
  `sigmoid_fast_block`/`tanh_fast_block` were hardcoded to `nn::Tensor` (==
  `TensorImpl<nn::Backend>`), so `LSTMLayerImpl<Backend>` silently failed to
  compile for any backend other than whichever one was currently selected.
  Fixed by templating them on `Backend` (`nn::TensorImpl<Backend>`); existing
  call sites deduce `Backend` from their argument, unchanged.
- `DeviceTensorBackend` (the documentation skeleton backend, see
  [Tensor](../Core/Tensor.md)) was missing several `TensorBackendParityContract`
  methods (`divide`, `add_col_vector_to_rows_inplace`, all ten `compare_*`/
  `compare_*_scalar` variants, `clamp`/`clamp_inplace`, `mean()`, the four
  `random(...)` overloads) — added, all delegating to its `m_host`
  `XTensorBackend` mirror like every other method in that file.
`Conv1dImpl`/`Conv2dImpl` also needed explicit template instantiation for
every concrete backend (not just `nn::Backend`) added in
`Conv1d_impl.cpp`/`Conv2d_impl.cpp`/`Conv2d_utils.cpp` — unlike header-only
layers (Linear, activations, LSTM, LifBPTT, losses, pooling), their method
bodies live in a separate `.cpp`, so other backends need their own explicit
instantiation to be linkable at all.

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

> **GPU test concurrency.** `pytorch_parity_gtest` instantiates
> `SYCLTensorBackend` directly, so it carries a shared CTest `RESOURCE_LOCK`
> (`cmake/GpuTestSerialization.cmake`) that serializes it against every other
> SYCL-touching test — this exists because concurrent kernel submission under
> AdaptiveCpp's HIP backend hung this dev machine hard enough to need a
> reboot. The lock does **not** apply to OpenCL: a full-suite freeze initially
> attributed to concurrent OpenCL kernel submission turned out to be residual
> fallout from that same SYCL/HIP incident — `ctest -j$(nproc)` under
> `NN_BACKEND=OpenCL` re-ran clean afterward. See
> [Tensor](../Core/Tensor.md#concurrent-gpu-test-serialization-2026-07-15-revised-same-day).

### Regenerate / extend

```bash
software/nn/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
software/nn/.venv/bin/python -m pip install snntorch
software/nn/.venv/bin/python software/nn/scripts/testing/gen_pytorch_refs.py
ctest --test-dir out/build/max-performance -R PyTorchParity   # XTensor/OpenCL/Device
# SYCL needs its own preset + the safety override (see cmake/SyclGpuCapabilityCheck.cmake):
cmake --preset=max-performance-sycl -DNN_SYCL_ACKNOWLEDGE_UNSUPPORTED_GPU=ON
cmake --build out/build/max-performance-sycl --target pytorch_parity_gtest -j$(nproc)
./out/build/max-performance-sycl/src/core/tensor/tests/pytorch_parity_gtest  # run single-threaded
```

Add a layer: add a case block in `gen_pytorch_refs.py` (save input/weights/output/
grads), add a `TYPED_TEST(PyTorchParityTyped, <Layer>)` in `pytorch_parity_gtest.cpp`
using `nn::<Layer>Impl<B>` (the template, not the `nn::<Layer>` alias tied to the
single selected `nn::Backend`), regenerate.

> **Backend gotcha.** The tests run on **all four backends**: xtensor
> (row-major), OpenCL (column-major), Device (row-major, host mirror), and
> SYCL (row-major, host mirror + optional device dispatch). The linear `at(k)`
> accessor exposes storage order, so a tensor filled from row-major fixture
> data via `at(k)` is transposed on OpenCL — which silently broke every
> structure-dependent op (matmul, conv, pool, LIF) while leaving elementwise
> ops (layout-invariant) passing. Always use the structured accessors
> `at(i,j)` / `at(i,j,k[,l])`, which address the same logical element on every
> backend; the test helpers enforce this.

See `software/nn/scripts/testing/README.md` for the full contract.

---

## Experiment05 smoke tests

Runtime errors (empty folds, shape mismatches, degenerate data) only appear when a
profile actually **runs**. The smoke suite runs every profile end-to-end with tiny
parameters so those surface quickly.

### Layout

`src/experiments/thesis/profiles/smoke/` mirrors every real profile
(`debug.json`, `phase00/*`, `phase01/*` → 315 total) keeping every code-path
selector (strategy, wavelet, scale, cepstral, modality, fusion_mode,
classifier.type, nested_cv, standardize_features) but shrinking run parameters:
`repeats=1`, `epochs=2`, `k_folds=2`, `samples_per_batch=4`, `max_samples=60`
(phase00) / `120` (phase01), `results_dir=results/thesis/smoke`.

### Automated mirroring (CMake)

The `thesis_smoke_profiles` target (in `ALL`) regenerates the mirror whenever any
source profile or `thesis_make_smoke_profiles.py` changes — a normal `cmake --build`
keeps `profiles/smoke/` in sync. `GLOB_RECURSE ... CONFIGURE_DEPENDS` picks up
added/removed profiles; the generator is stdlib-only (any `python3`, no torch).

### Running

```bash
cmake --build out/build/max-performance --target thesis -j$(nproc)
./scripts/testing/run_thesis_smoke.sh [phase00|phase01|all]   # needs the dataset present
```

The runner reports `PASS`/`FAIL` per profile with the captured error line and a
summary. ~315 runs — run on the data machine, without a short timeout.

### Bug it caught

Smoke testing the (previously never-run) DSNN phase01 profiles surfaced
`GroupKFoldPolicy: number of unique groups is less than n_splits`. Root cause:
`max_samples` truncation was `resize(first N)`, but samples are stored
**subject-contiguous** (~130 trials/speaker), so a capped run loaded only 2–3
speakers and nested CV's inner GroupKFold got fewer groups than folds. Fixed in
`ThesisDataset.cpp`: truncation now **round-robins across subjects**, so a capped set
spans every speaker (also fixes `debug.json` and any capped run).

---

## Related

- [Test Quality and Determinism](./Test-Quality-and-Determinism.md)
- [Experiment05](../Experiments/Thesis.md)
- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md)
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md)
