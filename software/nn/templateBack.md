# Plan: Refactor `Tensor` to use Template-based Backend

The goal is to replace the runtime polymorphism of `ITensorBackend` with compile-time templates. This removes virtual function overhead and enables inlining of critical paths (like `at()`), significantly improving performance in tight loops (e.g., SNN simulation).

## Architecture Changes

1.  **Template-based Tensor**:
    *   Rename `class Tensor` to `template <typename Backend> class TensorImpl`.
    *   Replace `std::unique_ptr<ITensorBackend> m_backend` with a value member `Backend m_backend`.
    *   Define a default alias: `using Tensor = TensorImpl<EigenTensorBackend>;`.
    *   This preserves the `nn::Tensor` API while allowing specific optimized backends.

2.  **Move Backend to Headers**:
    *   To allow `TensorImpl<EigenTensorBackend>` to be instantiated in user code, the definition of `EigenTensorBackend` must be visible.
    *   **Action**: Move `src/core/tensor/EigenTensorBackend.hpp` to `include/nn/tensor/EigenTensorBackend.hpp`.
    *   **Action**: Inline hot methods (`at()`, `rows()`, `cols()`) into `EigenTensorBackend.hpp` to fully realize the performance gains.

3.  **Remove Virtual Overhead**:
    *   `EigenTensorBackend` no longer *needs* to inherit from `ITensorBackend` (virtual dispatch isn't used), but can keep it for interface compliance if desired. However, for maximum speed, we want direct calls.
    *   Since `TensorImpl` calls `m_backend.at()`, if `EigenTensorBackend::at()` is not virtual, it's a direct function call.

## Step-by-Step Execution

### Phase 1: File Structure & Visibility
1.  Move `src/core/tensor/EigenTensorBackend.hpp` -> `include/nn/tensor/EigenTensorBackend.hpp`.
2.  Update `src/core/tensor/EigenTensorBackend.cpp` to include the new header location.
3.  Update `src/core/tensor/CMakeLists.txt` to reflect the move.

### Phase 2: Inline Hot Backend Methods
1.  Move the implementation of simple accessors (`at`, `rows`, `cols`, `size`, `shape`) from `EigenTensorBackend.cpp` to `EigenTensorBackend.hpp`.
    *   This is crucial. Without this, we just remove the virtual call but still have a function call overhead. Inlining allows the compiler to optimize the loop entirely.

### Phase 3: Templatize `Tensor`
1.  Modify `include/nn/tensor/Tensor.hpp`:
    *   Include `nn/tensor/EigenTensorBackend.hpp`.
    *   definition: `template <typename Backend = EigenTensorBackend> class TensorImpl`.
    *   Member: `Backend m_backend;` (Value type).
    *   Update all methods to use `.` instead of `->`.
    *   Update constructors to initialize `m_backend` directly (bypassing `TensorBackendFactory` for the default case).
2.  Add `using Tensor = TensorImpl<EigenTensorBackend>;` at the end of the namespace.

### Phase 4: Cleanup & Factory
1.  Update `TensorBackendFactory` (if kept) to return unique_ptrs only if needed for legacy/testing, or retire it if `Tensor` now owns the backend types directly.
2.  Update `Tensor.cpp` (which will essentially become empty or move to `.hpp`).

## Expected Impact
*   **Performance**: Significant speedup in `Leaky` and `Linear` layers due to inlined element access.
*   **Compilation**: Slower compile times due to `Eigen` headers being included in `Tensor.hpp`.
*   **API**: Source-compatible for most users (`nn::Tensor` still works).
