# Test Quality and Determinism

This guide records the recent test-suite hardening work and defines quality criteria for future tests.

## Scope

- Deterministic test behavior (stable seeds, stable tolerances, no timing races)
- Strong oracles (exact expected behavior whenever feasible)
- Runtime-aware suites (fast feedback loops without weakening behavior checks)

## Recent Improvements (2026-05)

### Deterministic hardening completed

- Replaced multiple broad checks (`non-zero`, `isfinite`, wide range-only checks) with exact deterministic expectations in core data loading, windowing, wave, and model tests.
- Tightened seeded-random sampler tests to compare against reproduced seeded permutations.
- Upgraded selected gradient-flow and state-reset tests to exact expected values.

### Runtime optimization completed

- Reduced workload sizes in the slowest tests while preserving contract intent and deterministic assertions.
- Full project test runtime was reduced from approximately 590s to approximately 77.62s in the validated run.

## Current Status Snapshot

- Full suite validated: `695/695` tests passing.
- A heuristic scan still reports `439` weak-oracle style assertions (`EXPECT_TRUE/FALSE`, inequality-only checks), concentrated in a few hotspot files.

Top hotspot files from the latest scan:

1. `src/experiments/autoencoderRunner/tests/ProfileAndResults_gtest.cpp` (54)
2. `src/core/layers/tests/layers_gtest.cpp` (30)
3. `src/core/statistics/tests/statistics_gtest.cpp` (29)
4. `src/experiments/guayaquil/tests/ComparativeExperiment_gtest.cpp` (23)
5. `src/core/optimizers/tests/optimizers_gtest.cpp` (22)

### OpenCL Lif Integration Coverage Snapshot (2026-05-10)

Coverage was collected from an instrumented build (`NN_ENABLE_COVERAGE=ON`) after running:

- `build-coverage/src/core/tensor/tests/opencl_tensor_backend_gtest`

Focused `lcov --extract` results for recently modified files:

1. `include/layers/spiking/Lif.hpp`
- Line coverage: 56.5% (124 lines)
- Function coverage: 42.9% (7 functions)

2. `src/core/tensor/opencl/OpenCLTensorBackend.cpp`
- Line coverage: 39.1% (4319 lines)
- Function coverage: 66.7% (156 functions)

Focused aggregate:
- Line coverage: 39.6% (1760/4443)
- Function coverage: 65.6% (107/163)

## SOTA-Aligned Test Quality Criteria

These criteria are aligned with peer-reviewed testing literature and adapted to this codebase.

1. Prefer exact oracles over broad predicates.
   - Avoid assertions that only prove "something happened".
   - Prefer exact tensors, exact shapes, exact counts, exact deterministic sequences.

2. Keep tests deterministic by construction.
   - Use fixed seeds.
   - Avoid wall-clock assertions for correctness.
   - Keep floating-point assertions numerically justified and stable.

3. Validate behavior, not only existence.
   - `not empty` checks should be followed by content/contract checks.
   - `isfinite` checks should be complemented with expected magnitude/value relationships.

4. Use mutation-resistant assertions on critical logic.
   - Prefer assertions that would fail under small behavioral changes.
   - For key modules, add stronger checks around boundary and failure paths.

5. Preserve fast feedback.
   - Minimize test runtime while retaining meaningful assertions.
   - Prefer smaller deterministic fixtures to large stochastic workloads.

## What Is Acceptable To Keep

Some broad assertions remain valid when they express real contracts:

- Null/resource checks immediately after allocation or external API calls.
- Exception and error-message contract assertions.
- Guard-condition tests for unsupported inputs and defensive behavior.

## Prioritized Next Pass

1. Remove tautological assertions first (always-true/always-false patterns).
2. Refactor hotspot files listed above from weak to deterministic-or-exact oracles.
3. Add a lightweight mutation-testing lane for critical modules (layers, optimizers, data loaders).
4. Add a CI quality gate that tracks weak-oracle count trend and fails on regressions.

## Practical Assertion Upgrade Patterns

- Replace `EXPECT_GT(norm, 0.0f)` with exact expected tensor/value where closed form exists.
- Replace `EXPECT_FALSE(vec.empty())` with exact `size()` and representative value checks.
- Replace broad range checks with exact expected mapping from fixture inputs.
- Keep `EXPECT_NEAR` with rationale-based tolerance (precision + operation count).

## Strict Core Coverage Gate

To enforce the policy that core libraries must remain at full coverage, use an
instrumented build and run the dedicated gate target:

```bash
cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DNN_ENABLE_COVERAGE=ON
cmake --build build-coverage -j$(nproc)
cmake --build build-coverage --target coverage-core-100
```

Behavior:
- Runs `ctest` in the selected build directory.
- Captures coverage with `lcov`.
- Filters to core library implementation/header files and excludes test sources.
- Fails if any file is below 100% line or 100% function coverage.

### Measured status (2026-07-19) — the gate does **not** currently pass

The gate had never actually run. Two defects prevented it, both now fixed:

1. **The target pointed at a nonexistent script.** `DevAndAnalysisTargets.cmake` invoked
   `${CMAKE_SOURCE_DIR}/tools/check_core_coverage.py`, but there is no `tools/` directory —
   the script lives at `scripts/dev/check_core_coverage.py`. Invoking the target failed
   immediately with "can't open file". A configure-time `FATAL_ERROR` guard now catches a
   future move.
2. **The file filter matched nothing.** The lcov `--extract` used `*/include/nn/*`, but there
   is no `include/nn/` directory — public headers live directly under `include/`. So the gate
   measured only `src/core/` while claiming to cover the public headers too, and on lcov 2.x an
   `--extract` pattern matching zero files is a hard error (exit 25). The pattern is now
   anchored to the source root (`<root>/src/core/*`, `<root>/include/*`) — it must stay
   anchored, since a bare `*/include/*` also swallows `/usr/include` and `_deps/`.

With both fixed, the first real measurement over 148 core files is:

| Metric | Coverage |
|---|---|
| **Lines** | **67.6%** (10,094 / 14,922) |
| **Functions** | **84.8%** (1,565 / 1,845) |
| Files below the 100% gate | **87 of 148** |

So the answer to "is the codebase at 100% coverage?" is **no**, and the "strict 100% gate"
was aspirational rather than enforced. The largest gaps are concentrated, not diffuse:

| File | Lines | Why |
|---|---|---|
| `src/core/tensor/opencl/OpenCLTensorBackend.cpp` | 34.7% (1819/5236) | GPU paths need a device; largest single gap by far |
| `src/core/utility/progress.cpp` | 0.0% (0/208) | Progress bars are UI, never exercised by tests |
| `src/core/statistics/confusion_matrix.cpp` | 0.0% (0/59) | Untested module |
| `*Printer.cpp` (2 files) | 0.0% | Debug printers |
| `src/core/tensor/opencl/DeviceMemory.cpp` | 29.6% | GPU-only |

A large share is OpenCL/GPU code that cannot run in a CPU-only coverage build, plus
UI/printer code. Reaching a genuine 100% would require either a GPU-enabled coverage run or
an explicit, documented exclusion list for device-only and presentation code — a policy
decision, not just more tests.

### Reproducibility contract: results must not change independently of a run

**The rule.** The same profile with the same seed must produce byte-identical results. A
re-run may not move the numbers.

**Enforced by** `ThesisReproducibility.*` in
`src/experiments/thesis/tests/thesis_feature_extraction_gtest.cpp`:

| Test | Asserts |
|---|---|
| `AutoencoderFeaturesAreIdenticalAcrossRuns` | SNN-AE × {direct, latency, poisson}: same seed → identical features |
| `AnnAutoencoderFeaturesAreIdenticalAcrossRuns` | ANN-AE: same seed → identical features |
| `DifferentSeedChangesAutoencoderFeatures` | different seed → *different* features |

The third test is not redundant. Without it the first two could pass **vacuously** — if the
seed never reached the model, every run would be identical for the wrong reason and the suite
would still look green.

> **A trap worth knowing.** That control test must use **`ann-ae`, not `snn-ae`**. A spiking
> latent can sit entirely below (or above) `V_th` in a small config, making it degenerate — no
> initialisation changes it, so the control fails for a reason that has nothing to do with seed
> propagation. It did exactly that when first written with `snn-ae`+`direct`+T=1 on 6 samples.
> The dense autoencoder has no threshold, so its output must move when the weights move.

**Verified empirically** on the committed results (2026-07-19): re-running
`p00_hc_haar_lfcc_c1_eeg` produced files byte-identical to the committed ones for all three
repeats, and two consecutive SNN-AE runs produced identical `d_truth`/`d_penalized`.

**If this contract breaks**, the first suspect is a `std::random_device` fallback in an
initializer — see the section below for how that happened once already. `src/core/utility/batching.cpp`
also shuffles with an unseeded RNG, but it is **not** reachable from the thesis path; if it
ever becomes reachable it must take a seed first.

### Nondeterminism audit (2026-07-19)

A systematic sweep for anything that could make results move without the inputs moving.
Two real defects were found and fixed; the rest are clean by construction.

| # | Category | Finding |
|---|---|---|
| 1 | `std::random_device` seeding | Thesis path clean — every RNG is seeded. `create_batches()` in `src/core/utility/batching.cpp` shuffled unseeded; 🔴 **fixed** — see below. |
| 2 | Time-based seeding (`time()`, `chrono::now()`) | None. |
| 3 | Iterating `unordered_map`/`unordered_set` into output | None. |
| 4 | `std::filesystem::directory_iterator` | 🔴 **Fixed** — see below. |
| 5 | OpenMP float `reduction` (summation order) | None. |
| 6 | Non-stable `std::sort` with possible ties | 🔴 **Fixed** — see below. |
| 7 | Parallel loops accumulating into shared state | None. `waveletOperations.cpp` uses loop-local sums; `ThesisClassifiers.cpp` writes disjoint `candidates[ii]` slots. Both deterministic by construction. |
| 8 | Pointer/address-based ordering or hashing | None. |
| 9 | Uninitialised reads | None flagged by the compiler. |

**Fix 1 — unsorted directory enumeration** (`paraconsistentBaseline/phase00_data.cpp`).
`directory_iterator` yields entries in an order the standard leaves *unspecified*; in practice
it follows the filesystem's layout, so it can differ between machines and after files are
rewritten. That made the row order of the feature/label vectors machine-dependent. It does
**not** affect the paraconsistent scores (α/β are per-class min/max/overlap, so order-invariant)
nor the label mapping (`build_label_index` sorts), but it **does** affect training — sample
order is the order SGD sees. Now collected into a vector and `std::sort`ed first.

The other three `directory_iterator` sites were already safe: `SubjectDiscovery.cpp` sorts by
`subject_id` (and its sqlite branch uses `ORDER BY id ASC`), `GuayaquilDataset.cpp` and
`FsddLoader.cpp` both sort.

**Fix 2 — arbitrary tie-break in the winner sort** (`ThesisParaconsistent.cpp`).
`rank_feature_sets` sorted on `d_penalized` alone. `std::sort` is **not stable**, so two feature
sets scoring exactly the same were left in an unspecified relative order — and `scores[0]` is
what gets written to the summary as the run's best. An exact tie could therefore flip the
reported winner between runs. The comparator now falls back to the label, making the result a
pure function of the data.

This is not hypothetical: on EEG the bark/mel/lfcc scales scored bit-identically, and
"bark won for EEG" was nothing but a tie-break artefact ([D6](./Engineering-Fixes-Log.md)).
That axis has since been removed for EEG, but the ordering should not be left to chance.

Both fixes were verified behaviour-preserving: re-running `p00_hc_haar_lfcc_c1_eeg` after them
still reproduces the committed results byte-identically for all three repeats.

**Fix 3 — unseeded batch shuffle** (`utility/batching.cpp`, `create_batches`).
Initially assessed as unreachable from any experiment; that was wrong — `WaveletAETraining.cpp`
(the waveletAE experiment) calls it every epoch. Batch order is the order SGD sees, so an
unseeded shuffle changed the trained weights between runs on identical inputs.

`create_batches()` now takes an optional trailing seed, matching the initializers' convention:

```cpp
create_batches(inputs, targets, batch_size);              // unseeded (unchanged default)
create_batches(inputs, targets, batch_size, /*seed=*/42); // reproducible
```

The default stays unseeded so all 16 existing call sites keep compiling and behaving as before;
**any path whose results must be reproducible has to pass a seed.** `WaveletAETraining` now
passes `random_seed + epoch` — the epoch offset keeps each epoch's order distinct (a fixed seed
would replay the same order every epoch and defeat shuffling) while making the whole schedule a
pure function of the run seed.

Guarded by `UtilTest.Batching{WithSeedIsDeterministic, DifferentSeedsGiveDifferentOrder,
SeededShufflePreservesEverySampleExactlyOnce}`. The middle test is the anti-vacuity check: it
fails if the seed is accepted but ignored. The two remaining unseeded callers are demos
(`resnet_classifier_demo`, `autoencoder_leakyrelu`), which produce no tracked results.

### A flaky test blocked the full gate run — root cause: unseeded weight init (fixed)

`ThesisSnnAe.PoissonLatentIsNonDegenerate` failed intermittently with an all-zero latent
(`max_abs = 0`, `max_var = 0`). It first showed up as "fails in Debug/coverage, passes in
`max-performance`", which looked like an optimisation-level problem. **It was not.** Running
the same Release binary 25 times gave **19 passes / 6 failures** — the test was simply flaky
at roughly a 24% rate, and the Debug/Release split was two unlucky draws.

**Root cause.** `extract_features()` seeded the *spike frames* from the experiment seed but
never the *model weights*. `AutoencoderConfig::initializer_seed` was left `nullopt`, so
`xavierInitializer`/`kaimingSNNInitializer` fell back to `std::random_device` (both document
this in their headers). Every run drew different initial weights. When a draw left all
encoder neurons below `V_th = 0.2`, nothing ever spiked and the latent came out exactly zero
— the dead-latent / No-Spike Problem of
[D1](./Engineering-Fixes-Log.md). Experiment 04 already set `initializer_seed = run_seed`;
Thesis was the only path that did not.

**Fix.** `ae_cfg.initializer_seed = seed;` in `ThesisFeatureExtraction.cpp`. After it: 25/25
passes, and repeated runs produce byte-identical output.

**This was never only a test problem.** Autoencoder feature extraction was
**non-reproducible**: the same profile with the same seed produced different features each
run. The fingerprint is visible in the committed Phase 00 tables — across the 3 repeats,
`std_d_truth` is **exactly 0.0000 for all 184 handcrafted rows** (they train nothing) but
**nonzero for all 24 autoencoder rows** (up to 0.185). Part of that reported spread is
uncontrolled initialisation randomness rather than genuine experimental variance, so the
autoencoder rows should be regenerated before the σ column is quoted as a stability measure.

> **Re-run implication.** Phase 00's 24 autoencoder profiles should be re-run under the fix
> for their numbers to be reproducible. The 184 handcrafted profiles are unaffected — they
> train nothing, so they already reproduce bit-for-bit. See the
> [Re-run Runbook](./Re-run-Runbook.md).

## Backend numerical parity (XTensor vs OpenCL)

`backend_parity_gtest` (`src/core/tensor/tests/backend_parity_gtest.cpp`) guards
against the CPU (XTensor, row-major) and GPU (OpenCL, column-major) backends
silently diverging. Every test builds identical deterministic inputs on both
backends, runs the same operation through the shared `TensorImpl` / layer
templates, and compares element-by-element via the backend-agnostic `at(i, j)`:

- elementwise ops, matmul family, reductions, slicing/reshape
- `Linear` forward/backward (incl. weight/bias gradients)
- `Lif` batched multi-step state evolution + backward (R/C/V_th gradients) —
  pits the OpenCL `lif_step_inplace` fast path against the XTensor generic path
- `LifIntegrator` forward/backward
- SNN-autoencoder-shaped chains (`Linear→Lif→Linear→LifIntegrator`), both with
  intermediate comparisons and with **zero intermediate host reads**

The no-host-reads variants matter: `at()` forces a device→host sync that can
mask stale-buffer bugs. Two real defects were found and fixed by this suite:

1. **Lazy-sync coherence** — binary/scalar/compare/transpose ops uploaded host
   `data_ptr()` without `sync_gpu_if_needed()`, so a GPU-resident operand (e.g.
   a fused `Linear` output) fed stale host data into the next op. All such ops
   now sync both operands on entry (no-op when already coherent).
2. **`reshape` semantics** — the OpenCL backend swapped shape metadata only,
   reinterpreting its column-major buffer; XTensor reshapes in row-major order.
   OpenCL `reshape` now physically permutes the buffer to preserve row-major
   logical order (the backend-parity contract).

The suite skips gracefully when no OpenCL device is present. Tolerances:
`2e-4` per-op, `5e-4` for chained results (GPU fp32 rounding accumulates).

## References

See [../References.md](../References.md), section "Software Testing and Test Quality".
