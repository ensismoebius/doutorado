# Tensor Migration Plan

## Scope and Objectives

- Remove direct Eigen usage outside backend implementations.
- Enforce Tensor + ITensorBackend as the only numerical surface.
- Introduce a mock backend to enable backend-agnostic tests.
- Keep the build green at each step; migrations must be reversible.

## Stepwise Plan (ordered by dependency/risk)

1. **Backend Guard Rails**
   - **Files**: cmake/EigenBan.hpp (new), cmake/EigenBan.cmake (new), scripts/check_eigen_leaks.py (new), eigen_allowlist.txt (new).
   - **Action**: Add compile-time guard toggle and a static leak checker that fails when Eigen appears outside the allowlist. Wire guard as an interface target; enable on tests first, then on core targets as they get migrated.
   - **Validation**: `cmake --build build --target check_eigen_leaks`; ensure all targets still compile with guard enabled for backends.

2. **Tensor Interface Detangling**
   - **Files**: src/core/tensor/Tensor.hpp/cpp, ITensorBackend.hpp, TensorBackendFactory.\*
   - **Action**: Remove Eigen includes from public headers; replace Eigen-specific setters/getters with backend-neutral APIs. Keep legacy Eigen access behind an opt-in shim if needed for staged migration.
   - **Validation**: tensor unit tests pass with both EigenTensorBackend and MockTensorBackend; no Eigen include required by Tensor.hpp consumers.

3. **MockTensorBackend Introduction**
   - **Files**: src/core/tensor/tests/MockTensorBackend.hpp/.cpp
   - **Action**: Implement a deterministic, Eigen-free backend with explicit shape checks and call logging. Provide helpers to introspect/clear logs in tests.
   - **Validation**: New mock-focused tests pass; mock exercises all ITensorBackend methods and matches algebraic expectations for small tensors.

4. **Initializers and Losses**
   - **Files**: src/core/initializers/\*, src/core/layers/MSELoss.hpp, CrossEntropyLoss.hpp, SpikeCountLoss.hpp
   - **Action**: Replace Eigen math with Tensor ops or backend calls; remove Eigen headers. Add mock-backed unit tests to verify shapes/values.
   - **Validation**: initializers_gtest and loss-specific tests run using MockTensorBackend; compile succeeds without Eigen in these translation units.

5. **Simple Layers (ReLU/LeakyReLU/Linear)**
   - **Files**: src/core/layers/Linear.hpp, ReLU.hpp, LeakyReLU.hpp
   - **Action**: Rewrite forward/backward to use Tensor methods only (matmul, add, scalar ops). Ensure bias broadcasting is backend-agnostic.
   - **Validation**: layers_gtest augmented with mock-based coverage that inspects backend call order; no `get_data_ref()` usage remains.

6. **Convolution/Pooling Layers**
   - **Files**: Conv2d.hpp/Conv2d_impl.cpp/Conv2d_utils.cpp, MaxPool2d.hpp
   - **Action**: Abstract convolution/pooling math through backend operations or dedicated tensor utilities. Remove Eigen maps.
   - **Validation**: Conv/pool tests run with mock backend for small kernels; shapes and padding validated.

7. **Optimizers**
   - **Files**: optimizers/Adam.hpp, SGD.hpp
   - **Action**: Use Tensor grad/data accessors that are backend-neutral; avoid Eigen maps. Add mock-backed tests for step() updates.
   - **Validation**: optimizers_gtest executes with MockTensorBackend; gradients and parameter updates match expected scalars.

8. **DataLoaders and Utilities**
   - **Files**: Dataset.hpp, mat_file_utils.\*, synthetic_spike_data.hpp, Normalization.hpp, parallel_batching.hpp
   - **Action**: Replace Eigen matrix manipulations with Tensor utilities; encapsulate file I/O to return Tensor instances or plain vectors before backend conversion.
   - **Validation**: dataLoader\_\*\_gtest run without Eigen includes; mock backend used to validate batching/shuffling order.

9. **Wave/Wavelet and Audio Features**
   - **Files**: wave/audioFeatureExtraction.cpp, wavelet/\*
   - **Action**: Move intermediate math to Tensor helpers; ensure resampling/filtering steps convert to backend tensors at boundaries.
   - **Validation**: Audio/EEG feature tests run without Eigen headers; numerical parity checked against reference arrays.

10. **Serialization and Saver**
    - **Files**: saver/NnSaver.hpp, NetworkSerializer.hpp, tests
    - **Action**: Ensure serialization uses backend-neutral tensor accessors; remove Eigen serialization paths.
    - **Validation**: NetworkSerializer_gtest passes with mock backend; round-trip matches bitwise for mock data.

11. **Experiments and Demos**
    - **Files**: src/experiments/00/_, demos/_
    - **Action**: Swap Eigen usage for Tensor APIs; adjust helper utilities accordingly.
    - **Validation**: Demo builds succeed without direct Eigen includes; runtime smoke tests still work with Eigen backend.

12. **Tighten Guard Rails**
    - **Files**: eigen_allowlist.txt (shrink), cmake/EigenBan.cmake
    - **Action**: Remove migrated files from allowlist; enable guard on all targets (no NN_ALLOW_EIGEN outside backends). Keep leak checker in CI.
    - **Validation**: Full build and tests pass with guard enabled; static check reports zero Eigen hits outside backend implementations.

## Reversibility Notes

- Each step touches a minimal file set; revertable via `git checkout <files>` for that step.
- Allowlist-based leak checker permits staged removal: remove files from allowlist only after the corresponding step is green.
- Mock backend tests do not alter production code paths and can be disabled by dropping the test target if needed.

## Testing Matrix per Step

- `ctest` (all GTests) on both EigenTensorBackend and MockTensorBackend where applicable.
- `cmake --build build --target check_eigen_leaks` to ensure no new Eigen usages.
- Optional sanitizer runs on layers/optimizers once Tensor interface is Eigen-free.

## Risks and Mitigations

- **API Surface**: Removing Eigen types from Tensor.hpp is a breaking change—mitigate with a temporary shim header and clear changelog entries.
- **Performance Regressions**: Naive backend implementations may be slower; add microbenchmarks after migration to confirm.
- **Broadcast Semantics**: Bias handling needs explicit tests to avoid silent shape misuse across backends.
- **DataLoader Semantics**: Conversions from Eigen matrices to tensors may alter memory layout; add golden-file tests for MAT readers.

## MockTensorBackend (Test Backend)

- **Storage**: owns `std::vector<float> m_data` in row-major order with explicit `m_shape`; gradients reuse a nested `MockTensorBackend` for parity with EigenTensorBackend.
- **Determinism**: naive loops, no external math libraries; stable across runs.
- **Logging**: every public method pushes a short string (e.g., `matmul:[2x3]x[3x1]`, `add_scalar`, `slice:n`) to `m_calls` for assertion.
- **Injection Example**: `auto b = std::make_unique<MockTensorBackend>(shape, data); auto* raw = b.get(); Tensor t(std::move(b)); auto out = t.matmul(other); EXPECT_FALSE(raw->log().empty());` — tests assert on `raw->log()` while validating outputs via `norm()`/`mean_squared_error` to avoid backend peeking.
