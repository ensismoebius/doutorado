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

1. `src/experiments/03/tests/ProfileAndResults_gtest.cpp` (54)
2. `src/core/layers/tests/layers_gtest.cpp` (30)
3. `src/core/statistics/tests/statistics_gtest.cpp` (29)
4. `src/experiments/04/tests/ComparativeExperiment_gtest.cpp` (23)
5. `src/core/optimizers/tests/optimizers_gtest.cpp` (22)

### OpenCL Leaky Integration Coverage Snapshot (2026-05-10)

Coverage was collected from an instrumented build (`NN_ENABLE_COVERAGE=ON`) after running:

- `build-coverage/src/core/tensor/tests/opencl_tensor_backend_gtest`

Focused `lcov --extract` results for recently modified files:

1. `include/nn/layers/spiking/Leaky.hpp`
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

## References

See [../References.md](../References.md), section "Software Testing and Test Quality".
