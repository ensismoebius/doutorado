# XTensorBackend Optimization Guide

This guide details the incremental optimization of `include/nn/tensor/xtensor/XTensorBackend.hpp`. 
**Execution Rule:** Apply exactly one sub-step at a time. After each sub-step, run the specified verification tests. Do not proceed if tests fail.

---

## Phase 1: Low-Hanging Fruit (Configuration & Basics)

### Step 1 — Enable SIMD via `XTENSOR_USE_XSIMD`
**Action:** 
In the `CMakeLists.txt` that builds `nn_xtensor_backend`, add:
```cmake
target_compile_definitions(nn_xtensor_backend PUBLIC XTENSOR_USE_XSIMD)
```
**Verification:**
1. Run `tensor_gtest`.
2. Add a temporary `static_assert(xsimd::batch<float>::size > 1, "SIMD not active");` in `XTensorBackend.hpp` to confirm.

### Step 2 — Use `xt::noalias` for in-place assignments
**Action:**
Update the following methods to use `xt::noalias(m_data) = ...` to avoid temporary allocations:
- `sqrt_inplace()`
- `square_inplace()`
- `clamp_inplace()`

**Verification:**
1. Run `tensor_gtest` (specifically tests covering `sqrt`, `square`, and `clamp`).

---

## Phase 2: Reducing Allocations in Hot Paths

### Step 3 — Vectorized Reductions
**Sub-step 3.1:** Replace manual loops in `sum_rows()` with `xt::sum` + `reshape`.
**Sub-step 3.2:** Replace manual loops in `sum_cols()` with `xt::sum` + `reshape`.

**Verification:**
1. Run `tensor_gtest` (verify `sum_rows` and `sum_cols` accuracy).

### Step 4 — Allocation-free Shape Comparison
**Sub-step 4.1:** Implement `bool same_shape(const XTensorBackend& other) const noexcept` using `m_data.shape()` (which returns a reference).
**Sub-step 4.2:** Replace `shape() != other.shape()` with `!same_shape(other)` in `add()`.
**Sub-step 4.3:** Replace in `subtract()`.
**Sub-step 4.4:** Replace in `multiply()`.
**Sub-step 4.5:** Replace in `matmul()`.

**Verification:**
1. Run `tensor_gtest` after each sub-step.

### Step 5 — Single-pass `row()` Evaluation
**Action:** 
Rewrite `row(Index i)` to use `xt::eval` combined with `xt::reshape_view` to collapse the view and reshape into a single allocation.

**Verification:**
1. Run `tensor_gtest` (verify `row()` return values and shapes).

---

## Phase 3: Algorithmic & Memory Optimizations

### Step 6 — Avoid Transpose Copies (`matmul_transposed`)
**Sub-step 6.1:** Implement `matmul_transposed(const XTensorBackend& other)` in `XTensorBackend` using `xt::linalg::dot` with the appropriate transpose flag (or by passing the transpose view directly to BLAS).
**Sub-step 6.2:** Audit the codebase for `A.matmul(B.transpose())` and replace with `A.matmul_transposed(B)`.

**Verification:**
1. Run `core_gtest` (this affects layers like `Linear` and `LSTM`).

### Step 7 — Fast Unchecked Accessors
**Sub-step 7.1:** Implement `at_unsafe(Index i)` and `at_unsafe(Index r, Index c)` as `noexcept` methods returning references to `m_data.data()`.
**Sub-step 7.2:** Replace `at()` with `at_unsafe()` inside `XTensorBackend`'s own methods.
**Sub-step 7.3:** Replace `at()` with `at_unsafe()` in `LSTMLayerImpl`'s inner loops (e.g., slice copying).

**Verification:**
1. Run `tensor_gtest` (after 7.2).
2. Run `core_gtest` (after 7.3).

### Step 8 — Flatten Gradient Storage
**Sub-step 8.1:** Change `m_grad_backend` from `unique_ptr<XTensorBackend>` to `std::optional<xt::xarray<float>>`.
**Sub-step 8.2:** Update `set_grad()`, `zero_grad()`, and `get_grad()` to work with the `optional` xarray.
**Sub-step 8.3:** Implement a lightweight `grad_ref()` wrapper to maintain API compatibility for layers that expect `XTensorBackend&`.

**Verification:**
1. Run `tensor_gtest` (verify gradient setting and retrieval).

---

## Phase 4: Final Polishing

### Step 9 — Use Fixed-Rank Temporaries
**Action:** 
In methods where the rank is known to be 2 (e.g., `sum_rows`, `sum_cols`), replace temporary `xt::xarray<float>` with `xt::xtensor<float, 2>`.

**Verification:**
1. Run `tensor_gtest`.

### Step 10 — BLAS Linkage Audit
**Action:** 
Run `ldd` on a compiled experiment binary (e.g., `experiment04`) to ensure it is linked against a BLAS provider (OpenBLAS, MKL, etc.).
```bash
ldd out/build/max-performance/src/experiments/04/experiment04 | grep -i blas
```
If missing, add `find_package(BLAS REQUIRED)` and link it to `nn_xtensor_backend`.

**Verification:**
1. Confirm linkage via `ldd`.
2. Run `tensor_gtest` to ensure stability.

---

## Summary Table

| Step | Target | Effort | Expected Gain | Verification |
|------|--------|--------|---------------|--------------|
| 1 | SIMD Config | Low | 4–8× element-wise | `tensor_gtest` |
| 2 | `noalias` | Trivial | $\downarrow$ Allocations | `tensor_gtest` |
| 3 | `xt::sum` | Low | $\uparrow$ Vectorization | `tensor_gtest` |
| 4 | `same_shape` | Low | $\downarrow$ Hot-path allocs | `tensor_gtest` |
| 5 | `row()` eval | Low | $\downarrow$ Row-slice allocs | `tensor_gtest` |
| 6 | `matmul_trans` | Medium | $\downarrow$ Transpose temps | `core_gtest` |
| 7 | `at_unsafe` | Low | $\downarrow$ Branching | `core_gtest` |
| 8 | Grad Flatten | Medium | $\downarrow$ Grad allocs | `tensor_gtest` |
| 9 | `xtensor<T,2>` | Medium | $\downarrow$ Shape allocs | `tensor_gtest` |
| 10 | BLAS Audit | Low | $\uparrow$ Matmul Speed | `ldd` |
