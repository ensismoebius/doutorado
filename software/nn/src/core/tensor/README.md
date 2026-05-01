# tensor

Purpose
- Core tensor abstraction and xtensor-backed implementations used across the codebase.

Source layout
- `src/core/tensor/opencl/`: OpenCL-specific tensor runtime and backend implementation.
- `src/core/tensor/eigen/`: Eigen-specific implementation files (reserved for backend-scoped code).
- `src/core/tensor/tests/`: Unit tests for tensor backends and tensor-level behavior.

Usage
- Use `nn::Tensor` for numeric data and call high-level operations from the `tensor` API. Prefer high-level helpers instead of dealing with raw Eigen matrices directly.

Integration
- The tensor module is a foundational dependency for layers, optimizers, and serialization. Link against the `tensor`/`linearAlgebra` targets as required.

Tests
- See `src/core/tensor/tests/` for construction, operations, and gradient examples.

Recent updates
- Fixed layer aggregator template substitution by correcting placeholder tokens in [cmake/Layers.hpp.in], then regenerating [include/nn/layers/Layers.hpp] during CMake configure to remove unresolved `@...@` build breakers.
- Added SIMD-assisted bulk fill paths in `XTensorBackend` for random initialization and constant fills (`fill`, `set_zero`, `set_ones`) through xsimd batch stores.
- Replaced `XTensorBackend::matmul_transposed` with a direct CBLAS `sgemm` path so CPU affine-style workloads no longer pay transpose-expression overhead before GEMM.
- Refactored `OpenCLTensorBackend` queue synchronization to a centralized `finish_queue_if_not_batching(...)` helper and replaced direct `clFinish(ctx.get_queue())` calls in kernel paths, so batch mode can amortize GPU synchronization overhead.
- Added RAII `OpenCLContext::BatchScope` and wired dense `Linear` forward/backward chains to batch OpenCL tensor operations through a single synchronization boundary.
- Extended `OpenCLTensorBackend` GPU-resident execution so `matmul` outputs can stay on device across dense affine chains, `add_col_vector_to_rows_inplace` can update resident buffers in place, and host storage is synchronized lazily on first CPU access.
- Added `matmul_rhs_transposed_kernel` in the OpenCL linear algebra program and routed `OpenCLTensorBackend::matmul_transposed` to this direct path, removing the extra transpose materialization previously used by dense and recurrent affine chains.
- Added standalone benchmark executable [src/core/tensor/tests/tensor_perf_bench.cpp](src/core/tensor/tests/tensor_perf_bench.cpp) to emit CSV timing for xtensor and OpenCL tensor workloads without making performance checks part of CTest.
- Rebuilt affected targets (`experiment03_lib`, `core_gtest`) and re-ran full CTest: 680/680 passed after these changes.
- Added backend primitive for rowwise addition of a `(cols, 1)` bias vector across all rows (`add_col_vector_to_rows_inplace`), exposed through `nn::Tensor`.
- Added OpenCL execution path in `OpenCLTensorBackend` for hot operations (`matmul`, `transpose`, `add`, `multiply`, `exp`, scalar add/multiply), with explicit runtime CPU fallback on kernel/context errors.
- Added ASan-safe runtime guard: OpenCL execution is disabled under AddressSanitizer builds to avoid third-party OpenCL runtime leak noise while preserving functional CPU fallback.
- Added OpenCL backend unit coverage in `src/core/tensor/tests/opencl_tensor_backend_gtest.cpp` for correctness invariants independent of device availability.
- OpenCL backend startup checks were centralized in `OpenCLTensorBackend::initialize_runtime_or_throw(bool)`, making sanitizer/runtime availability validation and profiling setup backend-owned rather than experiment-owned.
- Reorganized tensor backend source files into backend-specific folders under
	`src/core/tensor/opencl/` and `src/core/tensor/eigen/` for clearer ownership.
- OpenCL usage activity verification probe (`gpu_busy_percent` + reconstruction-MSE workload) was moved into `OpenCLTensorBackend::verify_runtime_activity_or_throw(...)`, removing probe-specific logic from experiment drivers.
- OpenCL runtime lifecycle now offers backend-owned RAII via
	`OpenCLTensorBackend::start_runtime_scope_or_throw(...)` and `RuntimeScope`,
	so experiment code no longer manages OpenCL buffer-pool shutdown directly.

Optimization techniques and references
- Backend-level fused rowwise update: move broadcast-add into the xtensor-backed kernel path to leverage vectorized matrix expressions and avoid per-element scalar loops at call sites (see [1], [2]).

Bibliographic references
- [1] Gene H. Golub and Charles F. Van Loan. Matrix Computations (4th ed.). Johns Hopkins University Press, 2013.
- [2] Intel 64 and IA-32 Architectures Optimization Reference Manual. Intel Corporation, current edition.
