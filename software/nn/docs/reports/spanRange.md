# C++20 Refactoring Plan: `std::span` and `std::ranges`

## 1. Candidate Loops for `std::ranges`

### `src/core/linearAlgebra/linearAlgebra.cpp`

#### `dotProduct` (Line ~50)
*   **Current**: Index-based `for` loop accumulating product.
*   **Proposed**: `std::transform_reduce` (header `<numeric>`).
*   **Impact**: clear intent, potential for vectorization (unsequenced policy).

#### `normalizeVectorToSum1` (Line ~90)
*   **Current**: Two index-based loops (sum, then divide).
*   **Proposed**:
    1. `std::accumulate` (or `std::reduce`) for sum.
    2. `std::ranges::for_each` with lambda `[&](double& val) { val /= sum; }` for division.
*   **Impact**: Removes manual indexing, prevents off-by-one errors.

## 2. Candidate `std::vector` Usages for `std::span`

### `src/core/tensor/ITensorBackend.hpp` (and implementations)

The core tensor interface relies heavily on `std::vector` for shape passing, forcing allocations even for fixed-size shapes passed as initializer lists.

*   **Signatures**:
    *   `virtual void construct(const std::vector<Index>& shape) = 0;`
    *   `virtual float& at(const std::vector<Index>& indices) = 0;`
    *   `virtual void reshape(const std::vector<Index>& new_shape) = 0;`
*   **Proposed**: Change `const std::vector<Index>&` to `std::span<const Index>`.
*   **Ownership**: Non-owning view. The implementations typically copy the shape into internal storage anyway.
*   **Safety**: Implementations immediately use the data. No lifetime issues.

### `src/core/linearAlgebra/linear_algebra.hpp`

*   **Signatures**:
    *   `double dotProduct(const std::vector<double>& a, const std::vector<double>& b);`
    *   `void normalizeVectorToSum1(double* signal, long signalLength);` (Legacy C-style)
*   **Proposed**:
    *   `dotProduct`: Accept `std::span<const double>`.
    *   `normalizeVectorToSum1`: Accept `std::span<double>` (replaces pointer+length).
    *   `calcOrthogonalVector`: Remove legacy raw pointer overload, ensure `span` overload is primary.

### `src/core/utility/batching.hpp`

*   **Signature**: `std::vector<Batch> create_batches(const std::vector<nn::Tensor>& inputSamples, const std::vector<nn::Tensor>& targets, ...)`
*   **Proposed**: Use `std::span<const nn::Tensor>` for inputs.
*   **Benefit**: Allows batching subsets of data without copying vectors.

## 3. Non-Candidates (Explicitly Excluded)

### `src/core/layers/Conv2d_utils.cpp`
*   **Loops**: `im2col_optimized`, `compute_indices`.
*   **Reason**: Explicit definition of OpenMP parallelism (`#pragma omp parallel for collapse(2)`). Replacing this with `std::ranges` would currently require non-standard parallel policies or lose the explicit threading model (thread-local buffer optimization in `compute_indices`).

### `src/core/linearAlgebra/linearAlgebra.cpp`
*   **Function**: `derivative`
*   **Reason**: The function resizes the input vector (`vector.resize`). `std::span` is non-resizing.

## 4. Refactoring Order

### Step 1: Linear Algebra Utilities [COMPLETED]
*   **Files**: `src/core/linearAlgebra/*`.
*   **Actions**:
    *   Update `dotProduct` to use `std::span` and `std::transform_reduce`.
    *   Refactor `normalizeVectorToSum1` to `std::span`.
    *   Delete legacy pointer-based `calcOrthogonalVector`.
*   **Verification**: Run `linearAlgebra_gtest`.

### Step 2: Core Tensor Interface [COMPLETED]
*   **Files**: `include/nn/tensor/ITensorBackend.hpp`, `src/core/tensor/EigenTensorBackend.*`, `src/core/tensor/tests/MockTensorBackend.*`.
*   **Actions**:
    *   Replace `const std::vector<Index>&` with `std::span<const Index>`.
*   **Verification**: Run `tensor_gtest`, `layers_gtest` (as layers depend on tensor).

### Step 3: Utility/Batching [COMPLETED]
*   **Files**: `src/core/utility/batching.*`.
*   **Actions**:
    *   Update `create_batches` signature.
*   **Verification**: Run `dataLoaders_gtest`.
