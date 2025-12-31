# Tensor Migration Guide - Step-by-Step Implementation

This guide provides a detailed, actionable plan for migrating the codebase from direct Eigen usage to exclusively using the `nn::Tensor` class. Each step includes specific code changes and validation.

## Phase 1: Implement Missing Tensor Methods

### Step 1.1: Add Row/Column Access Methods

**File**: `src/core/tensor/Tensor.hpp` and `src/core/tensor/Tensor.cpp`

**Add to Tensor.hpp**:

```cpp
// Row and column access
auto row(Eigen::Index i) const -> Tensor;
auto col(Eigen::Index j) const -> Tensor;
auto leftCols(Eigen::Index n) const -> Tensor;
auto topRows(Eigen::Index n) const -> Tensor;
```

**Add to Tensor.cpp**:

```cpp
auto Tensor::row(Eigen::Index i) const -> Tensor
{
    if (m_shape.size() != 2) {
        throw std::invalid_argument("row() only valid for 2D tensors");
    }
    if (i < 0 || i >= m_shape[0]) {
        throw std::out_of_range("Row index out of range");
    }
    Eigen::MatrixXf row_data = m_data.row(i);
    return Tensor(row_data);
}

auto Tensor::col(Eigen::Index j) const -> Tensor
{
    if (m_shape.size() != 2) {
        throw std::invalid_argument("col() only valid for 2D tensors");
    }
    if (j < 0 || j >= m_shape[1]) {
        throw std::out_of_range("Column index out of range");
    }
    Eigen::MatrixXf col_data = m_data.col(j);
    return Tensor(col_data);
}

auto Tensor::leftCols(Eigen::Index n) const -> Tensor
{
    if (m_shape.size() != 2) {
        throw std::invalid_argument("leftCols() only valid for 2D tensors");
    }
    if (n < 0 || n > m_shape[1]) {
        throw std::out_of_range("Invalid number of columns");
    }
    Eigen::MatrixXf cols_data = m_data.leftCols(n);
    return Tensor(cols_data);
}

auto Tensor::topRows(Eigen::Index n) const -> Tensor
{
    if (m_shape.size() != 2) {
        throw std::invalid_argument("topRows() only valid for 2D tensors");
    }
    if (n < 0 || n > m_shape[0]) {
        throw std::out_of_range("Invalid number of rows");
    }
    Eigen::MatrixXf rows_data = m_data.topRows(n);
    return Tensor(rows_data);
}
```

**Test**: Add to `tensor_gtest.cpp`:

```cpp
TEST(TensorTest, RowColAccess)
{
    nn::Tensor t(3, 4);
    t.at(0, 0) = 1.0f; t.at(0, 1) = 2.0f; t.at(0, 2) = 3.0f; t.at(0, 3) = 4.0f;
    t.at(1, 0) = 5.0f; t.at(1, 1) = 6.0f;

    auto row0 = t.row(0);
    ASSERT_EQ(row0.get_shape(), std::vector<Eigen::Index>({1, 4}));
    EXPECT_EQ(row0.at(0, 0), 1.0f);
    EXPECT_EQ(row0.at(0, 3), 4.0f);

    auto col1 = t.col(1);
    ASSERT_EQ(col1.get_shape(), std::vector<Eigen::Index>({3, 1}));
    EXPECT_EQ(col1.at(0, 0), 2.0f);
    EXPECT_EQ(col1.at(1, 0), 6.0f);

    auto left2 = t.leftCols(2);
    ASSERT_EQ(left2.get_shape(), std::vector<Eigen::Index>({3, 2}));
    EXPECT_EQ(left2.at(0, 0), 1.0f);
    EXPECT_EQ(left2.at(0, 1), 2.0f);
}
```

### Step 1.2: Add Block Operations

**Add to Tensor.hpp**:

```cpp
// Block operations
auto block(Eigen::Index row, Eigen::Index col, Eigen::Index rows, Eigen::Index cols) const -> Tensor;
void setBlock(Eigen::Index row, Eigen::Index col, const Tensor& block);
```

**Add to Tensor.cpp**:

```cpp
auto Tensor::block(Eigen::Index row, Eigen::Index col, Eigen::Index rows, Eigen::Index cols) const -> Tensor
{
    if (m_shape.size() != 2) {
        throw std::invalid_argument("block() only valid for 2D tensors");
    }
    if (row < 0 || col < 0 || rows < 0 || cols < 0 ||
        row + rows > m_shape[0] || col + cols > m_shape[1]) {
        throw std::out_of_range("Block dimensions out of range");
    }
    Eigen::MatrixXf block_data = m_data.block(row, col, rows, cols);
    return Tensor(block_data);
}

void Tensor::setBlock(Eigen::Index row, Eigen::Index col, const Tensor& block)
{
    if (m_shape.size() != 2 || block.get_shape().size() != 2) {
        throw std::invalid_argument("setBlock() only valid for 2D tensors");
    }
    if (row < 0 || col < 0 ||
        row + block.get_shape()[0] > m_shape[0] ||
        col + block.get_shape()[1] > m_shape[1]) {
        throw std::out_of_range("Block position out of range");
    }
    m_data.block(row, col, block.get_shape()[0], block.get_shape()[1]) = block.get_data_ref();
}
```

### Step 1.3: Add Element-wise Operations

**Add to Tensor.hpp**:

```cpp
// Element-wise operations
auto add(const Tensor& other) const -> Tensor;
auto multiply(const Tensor& other) const -> Tensor;
auto add_scalar(float scalar) -> Tensor&;
auto multiply_scalar(float scalar) -> Tensor&;
```

**Add to Tensor.cpp**:

```cpp
auto Tensor::add(const Tensor& other) const -> Tensor
{
    if (m_shape != other.get_shape()) {
        throw std::invalid_argument("Shape mismatch in add");
    }
    Eigen::MatrixXf result = m_data + other.get_data_ref();
    return Tensor(result);
}

auto Tensor::multiply(const Tensor& other) const -> Tensor
{
    if (m_shape != other.get_shape()) {
        throw std::invalid_argument("Shape mismatch in multiply");
    }
    Eigen::MatrixXf result = m_data.cwiseProduct(other.get_data_ref());
    return Tensor(result);
}

auto Tensor::add_scalar(float scalar) -> Tensor&
{
    m_data.array() += scalar;
    return *this;
}

auto Tensor::multiply_scalar(float scalar) -> Tensor&
{
    m_data.array() *= scalar;
    return *this;
}
```

### Step 1.4: Build and Test New Methods

**Command**:

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn/build
cmake --build . --target nn_tensor_gtest
ctest --test-dir . -R TensorTest
```

## Phase 2: Migrate Core Files

### Step 2.1: Update `src/core/statistics/multiClassMetrics.cpp`

**Current code**:

```cpp
Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(n_classes, n_classes);
```

**New code**:

```cpp
// For now, use float tensor (consider adding int tensor later)
nn::Tensor cm(n_classes, n_classes);
```

**Note**: This changes confusion matrix to float. If integer precision is needed, implement `IntTensor` class.

### Step 2.2: Update `src/experiments/02/experiment_02.cpp`

**Find and replace**:

```cpp
// Before
Eigen::MatrixXf x(1, train_features[i].size());
Eigen::MatrixXf y = Eigen::MatrixXf::Zero(1, n_classes);
Eigen::MatrixXf out_data = output.get_data_ref();

// After
nn::Tensor x(1, static_cast<Eigen::Index>(train_features[i].size()));
nn::Tensor y(1, n_classes);
nn::Tensor out_data = output;  // Direct assignment instead of get_data_ref()
```

**For test data**:

```cpp
// Before
Eigen::MatrixXf x(1, test_feat.size());

// After
nn::Tensor x(1, static_cast<Eigen::Index>(test_feat.size()));
```

### Step 2.3: Update `src/demos/exec_resnet_demo/resnet_demo.cpp`

**Find and replace**:

```cpp
// Before
Eigen::MatrixXf mat = std::move(*mat_opt);
Eigen::MatrixXf x = mat.row(i).leftCols(n_features);
Eigen::MatrixXf y = Eigen::MatrixXf::Zero(1, n_classes);

// After
nn::Tensor mat = nn::Tensor(std::move(*mat_opt));
nn::Tensor x = mat.row(i).leftCols(n_features);
nn::Tensor y(1, n_classes);
```

### Step 2.4: Update `src/experiments/00/phase00.cpp`

**Find and replace**:

```cpp
// Before
auto extract_wavelet_features_single_trial(const Eigen::MatrixXf& signal_data, //
Eigen::VectorXf channel_data = signal_data.row(channel_row_idx);

// After
auto extract_wavelet_features_single_trial(const nn::Tensor& signal_data, //
nn::Tensor channel_data = signal_data.row(channel_row_idx);
```

### Step 2.5: Update `src/core/utility/batching.cpp`

**Find and replace**:

```cpp
// Before
x_concat.block(j * input_rows, 0, input_rows, input_cols) = x_batch_vec[j].get_data_ref();
y_concat.block(j * target_rows, 0, target_rows, target_cols) = y_batch_vec[j].get_data_ref();

// After
x_concat.setBlock(j * input_rows, 0, x_batch_vec[j]);
y_concat.setBlock(j * target_rows, 0, y_batch_vec[j]);
```

**Note**: This assumes `x_concat` and `y_concat` are already `nn::Tensor` objects.

### Step 2.6: Update Layer Files

#### `src/core/layers/Conv2d_impl.cpp`

**Find and replace**:

```cpp
// Before
bg.col(0) = summed;
bg.row(0) = summed.transpose();

// After
// Assuming bg is nn::Tensor, need to implement these operations
// For now, use get_data_ref() but plan to add Tensor methods
bg.get_data_ref().col(0) = summed;
bg.get_data_ref().row(0) = summed.transpose();
```

#### `src/core/layers/Conv2d_utils.cpp`

**Find and replace**:

```cpp
// Before
bias_vector = b.col(0);
bias_vector = b.row(0).transpose();

// After
bias_vector = b.col(0);
bias_vector = b.row(0).transpose();  // These should work with new methods
```

**For array operations**:

```cpp
// Before
Eigen::Map<Eigen::VectorXf>(matrix.row(i).data(), num_cols).array() += bias_val;
matrix.row(r).array() += bias_vector(idx);

// After
// Need to implement element-wise operations on Tensor
// For now, use get_data_ref()
matrix.get_data_ref().row(i).segment(0, num_cols).array() += bias_val;
matrix.get_data_ref().row(r).array() += bias_vector.get_data_ref()(idx);
```

### Step 2.7: Update `src/core/wave/audioFeatureExtraction.cpp`

**Remove Eigen includes** and update any `Eigen::VectorXf` to `nn::Tensor`.

**Update function signatures** that use `Eigen::VectorXf` or `Eigen::MatrixXf`.

### Step 2.8: Build and Test Migration

**Command**:

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn/build
cmake --build . --target all
ctest --output-on-failure
```

## Phase 3: Clean Up and Finalize

### Step 3.1: Remove Unused Eigen Includes

**Files to update**:

- Remove `#include <Eigen/Dense>` from migrated files
- Keep only in `Tensor.hpp/cpp`

### Step 3.2: Update Documentation

- Update any code comments referencing Eigen
- Update function documentation

### Step 3.3: Final Testing

**Run full test suite**:

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn/build
ctest --output-on-failure -j$(nproc)
```

**Run specific experiments**:

```bash
# Test experiment 02
./bin/experiment_02

# Test resnet demo
./bin/resnet_demo
```

### Step 3.4: Performance Validation

**Benchmark key operations** to ensure no performance regression.

## Implementation Order

1. ✅ Implement missing Tensor methods (Phase 1)
2. 🔄 Migrate core experiment files (experiment_02.cpp, resnet_demo.cpp)
3. 🔄 Migrate utility files (batching.cpp, phase00.cpp)
4. 🔄 Migrate layer files (Conv2d\_\*.cpp)
5. 🔄 Migrate wave processing files
6. 🔄 Clean up includes and documentation
7. 🔄 Final testing and validation

## Rollback Commands

If issues arise, rollback specific changes:

```bash
git checkout HEAD~1 -- src/experiments/02/experiment_02.cpp
```

## Success Criteria

- ✅ All tests pass
- ✅ All executables build and run
- ✅ No direct Eigen::MatrixXf usage in application code
- ✅ Tensor class provides all needed functionality
- ✅ Performance is maintained or improved

---

_Follow this step-by-step guide to complete the migration systematically._

## Phase 1: Replace Direct Eigen Declarations

### Function Signature Updates

Many functions currently accept `Eigen::MatrixXf` or `Eigen::VectorXf` parameters. Update these to use `nn::Tensor`:

#### Examples:

- **Current**: `void process_data(const Eigen::MatrixXf& data)`
- **New**: `void process_data(const nn::Tensor& data)`

#### Files to Update:

- `src/experiments/00/phase00.cpp`: `extract_wavelet_features_single_trial`
- Any data loading functions
- Layer forward/backward methods (if they take raw matrices)

### Gradient Handling Considerations

When replacing Eigen operations, ensure gradient computations are preserved:

- **Current**: Direct Eigen operations don't track gradients
- **New**: Tensor operations should accumulate gradients properly
- **Action**: For custom operations, implement backward passes that update `tensor.get_grad_ref()`

Example:

```cpp
// Instead of: Eigen::MatrixXf result = a.array() * b.array();
// Use: Tensor result = element_wise_multiply(a, b); // with proper grad accumulation
```

### Core Files

#### `src/core/statistics/multiClassMetrics.cpp`

- **Current**: `Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(n_classes, n_classes);`
- **New**: Use `nn::Tensor` with integer type (may need to extend Tensor for int types or use float)
- **Note**: Tensor currently uses float. May need to add int support or use float for confusion matrix.

#### `src/core/wave/audioFeatureExtraction.cpp`

- **Current**: Includes Eigen for `Eigen::Map` and `Eigen::VectorXf`
- **New**: Replace `Eigen::VectorXf` with `nn::Tensor` where possible
- **Action**: Change function signatures and implementations to use Tensor

### Data Loading and I/O

Functions that load data from files (MAT, NPY, etc.) currently return `Eigen::MatrixXf`. Update to return `nn::Tensor`:

- **MatFile readers**: Change return types from `Eigen::MatrixXf` to `nn::Tensor`
- **Data loaders**: Update Dataset/DataLoader to work with Tensor
- **File I/O**: Ensure serialization methods work with Tensor

#### `src/experiments/02/experiment_02.cpp`

- **Current**:
  ```cpp
  Eigen::MatrixXf x(1, train_features[i].size());
  Eigen::MatrixXf y = Eigen::MatrixXf::Zero(1, n_classes);
  Eigen::MatrixXf out_data = output.get_data_ref();
  ```
- **New**:
  ```cpp
  nn::Tensor x(1, static_cast<Eigen::Index>(train_features[i].size()));
  nn::Tensor y(1, n_classes);
  nn::Tensor out_data = output; // or copy if needed
  ```

#### `src/experiments/00/phase00.cpp`

- **Current**:
  ```cpp
  auto extract_wavelet_features_single_trial(const Eigen::MatrixXf& signal_data, //
  Eigen::VectorXf channel_data = signal_data.row(channel_row_idx);
  ```
- **New**:
  ```cpp
  auto extract_wavelet_features_single_trial(const nn::Tensor& signal_data, //
  nn::Tensor channel_data = signal_data.slice({channel_row_idx}); // Need to implement row slicing
  ```

#### `src/demos/exec_resnet_demo/resnet_demo.cpp`

- **Current**:
  ```cpp
  Eigen::MatrixXf mat = std::move(*mat_opt);
  Eigen::MatrixXf x = mat.row(i).leftCols(n_features);
  Eigen::MatrixXf y = Eigen::MatrixXf::Zero(1, n_classes);
  ```
- **New**:
  ```cpp
  nn::Tensor mat = nn::Tensor(std::move(*mat_opt));
  nn::Tensor x = mat.slice({i}).slice_cols(0, n_features); // Need to add column slicing
  nn::Tensor y(1, n_classes);
  ```

### Layer Files

#### `src/core/layers/Conv2d_impl.cpp`

- **Current**: Uses `.col(0)`, `.row(0)` on Eigen matrices
- **New**: Implement equivalent operations on Tensor
- **Action**: Add Tensor methods for bias operations

#### `src/core/layers/Conv2d_utils.cpp`

- **Current**: Extensive Eigen operations like `.col(0)`, `.row(i)`, `.array() +=`
- **New**: Wrap in Tensor methods
- **Action**: Add bias addition methods to Tensor

#### `src/core/utility/batching.cpp`

- **Current**: `x_concat.block(...) = x_batch_vec[j].get_data_ref();`
- **New**: Use Tensor assignment methods
- **Action**: Add block assignment to Tensor

## Phase 2: Add Missing Tensor Methods

Based on usage analysis, add these methods to `nn::Tensor`:

### Essential Methods to Add

1. **Row/Column Operations**:
   - `Tensor row(Eigen::Index i) const`
   - `Tensor col(Eigen::Index j) const`
   - `Tensor rows(std::span<const int> indices) const` (extend slice)
   - `Tensor leftCols(Eigen::Index n) const`

2. **Block Operations**:
   - `void setBlock(Eigen::Index row, Eigen::Index col, const Tensor& block)`
   - `Tensor block(Eigen::Index row, Eigen::Index col, Eigen::Index rows, Eigen::Index cols) const`

3. **Element-wise Operations**:
   - `Tensor& array() += scalar` (or equivalent)
   - `Tensor& array() += vector`

4. **Type Conversions**:
   - Consider adding int tensor support or conversion methods

5. **Utility Methods**:
   - `Tensor transpose() const`
   - `Tensor sum() const` (for reductions)

### Implementation Notes

- For N-D tensors, some operations may only make sense for 2D
- Add dimension checks where appropriate
- Maintain gradient compatibility

## Phase 3: Update Includes and Dependencies

- Remove `#include <Eigen/Dense>` from files that no longer need direct Eigen access
- Keep Eigen include in `Tensor.hpp/cpp` as internal implementation
- Update any Eigen-specific code in CMakeLists.txt (if any)
- Ensure Eigen is still linked for Tensor's internal use
- Update header includes in affected files

## Phase 4: Testing and Validation

1. **Unit Tests**: Ensure all Tensor methods work correctly
2. **Integration Tests**: Run experiments and demos
3. **Performance**: Check for performance regressions
4. **Memory**: Ensure no memory leaks

## File-by-File Migration Checklist

### ✅ Phase 2: Core Experiment Files (Completed)

- [x] `src/experiments/02/experiment_02.cpp` - Migrated Eigen matrix operations to Tensor
- [x] `src/demos/exec_resnet_demo/resnet_demo.cpp` - Migrated data loading and row operations
- [x] `src/experiments/00/phase00.cpp` - Migrated function signatures and Eigen::Index usage
- [x] `src/core/utility/batching.cpp` - Migrated batch concatenation to Tensor operations
- [x] `src/core/statistics/multiClassMetrics.cpp` - Migrated confusion matrix to Tensor

### 🔄 Phase 5: Layer Files (In Progress)

- [x] `src/core/layers/Linear.hpp` - Migrated input_cache and matrix operations
- [x] `src/core/layers/ReLU.hpp` - Migrated gradient storage and activation operations
- [ ] `src/core/layers/Conv2d_impl.cpp` - Complex im2col transformation migration
- [ ] `src/core/layers/Conv2d_utils.cpp` - Bias operations and matrix manipulations

### Medium Priority (Remaining)

- [ ] `src/core/wave/audioFeatureExtraction.cpp` - FFT and signal processing operations
- [ ] Other activation layers (Sigmoid, Tanh, etc.)
- [ ] Loss function layers (MSELoss, CrossEntropyLoss, etc.)

### Low Priority

- [ ] Other demo files
- [ ] Test files (may keep some Eigen for testing internals)

## Phase 5: Remaining Work (Future Phases)

This phase covers the migration of layer files and other components that still contain direct Eigen usage. These files are more complex due to their mathematical operations and performance requirements.

### Step 5.1: Migrate Linear Layer

**File**: `src/core/layers/Linear.hpp` and `src/core/layers/Linear_impl.cpp`

**Current Issues**:

- Uses `Eigen::MatrixXf input_cache` for storing inputs
- Direct Eigen operations in forward/backward passes

**Migration Steps**:

1. **Update member variables**:

   ```cpp
   // Before
   Eigen::MatrixXf input_cache;

   // After
   nn::Tensor input_cache;
   ```

2. **Update forward method**:

   ```cpp
   // Before
   Eigen::MatrixXf const output = input.get_data_ref() * weight.get_data_ref().transpose() + bias.get_data_ref();

   // After
   // Need to implement matrix multiplication in Tensor
   // For now, use get_data_ref() but plan to add Tensor::matmul
   nn::Tensor output = nn::Tensor(input.get_data_ref() * weight.get_data_ref().transpose() + bias.get_data_ref());
   ```

3. **Update backward method**:

   ```cpp
   // Before
   Eigen::MatrixXf const grad_input = grad_previous.get_data_ref() * weight.get_data_ref();

   // After
   nn::Tensor grad_input = nn::Tensor(grad_previous.get_data_ref() * weight.get_data_ref());
   ```

**Note**: Linear layer requires matrix multiplication. Consider adding `Tensor::matmul(const Tensor& other)` method.

### Step 5.2: Migrate ReLU and Activation Layers

**File**: `src/core/layers/ReLU.hpp` and `src/core/layers/ReLU_impl.cpp`

**Current Issues**:

- Uses `Eigen::MatrixXf relu_grad` for gradient storage
- Element-wise operations using Eigen arrays

**Migration Steps**:

1. **Update member variables**:

   ```cpp
   // Before
   Eigen::MatrixXf relu_grad;

   // After
   nn::Tensor relu_grad;
   ```

2. **Update forward method**:

   ```cpp
   // Before
   Eigen::MatrixXf const activated = input.get_data_ref().array().max(0);

   // After
   // Need element-wise max with scalar
   nn::Tensor activated = nn::Tensor(input.get_data_ref().array().max(0));
   ```

3. **Update backward method**:

   ```cpp
   // Before
   Eigen::MatrixXf const grad_input = grad_output.get_data_ref().array() * relu_grad.array();

   // After
   // Need element-wise multiplication
   nn::Tensor grad_input = nn::Tensor(grad_output.get_data_ref().array() * relu_grad.get_data_ref().array());
   ```

**Required Tensor Methods**:

- `Tensor relu() const` - element-wise max with 0
- `Tensor relu_backward(const Tensor& grad_output, const Tensor& relu_mask) const`

### Step 5.3: Migrate Conv2d Layer (Complex)

**Files**: `src/core/layers/Conv2d.hpp`, `src/core/layers/Conv2d_impl.cpp`, `src/core/layers/Conv2d_utils.cpp`

**Current Issues**:

- Extensive use of Eigen for im2col transformation
- Complex matrix operations for convolution
- Bias addition operations

**Migration Steps**:

1. **Update Conv2d_impl.cpp forward pass**:

   ```cpp
   // Before: Direct Eigen Map usage
   Eigen::Map<const Eigen::MatrixXf> im2col_mapped(...);
   Eigen::Map<const Eigen::MatrixXf> weights_mapped(...);
   Eigen::MatrixXf output_2d(out_channels_, total_patch_cols);

   // After: Need Tensor equivalents
   // This is complex - may need to keep Eigen for im2col but wrap in Tensor
   ```

2. **Update bias operations**:

   ```cpp
   // Before
   Eigen::Map<Eigen::MatrixXf> weights_map(...);
   weights_map = Eigen::MatrixXf::Random(...) * stddev;

   // After
   // Need Tensor random initialization method
   weights_.random_normal(0.0f, stddev);
   ```

3. **Update backward pass**:

   ```cpp
   // Before: Complex Eigen operations
   Eigen::MatrixXf& weights_grad = weights_.get_grad_ref();

   // After: Use Tensor gradient operations
   nn::Tensor weights_grad = weights_.grad();
   ```

**Required Tensor Methods for Conv2d**:

- `Tensor im2col(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) const`
- `Tensor col2im(int input_h, int input_w, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) const`
- `void add_bias(const Tensor& bias)`
- `Tensor conv2d(const Tensor& input, const Tensor& weight, int stride, int padding)`

**Note**: Conv2d migration is the most complex. Consider implementing convolution operations at the Tensor level rather than migrating existing Eigen code.

### Step 5.4: Migrate Other Layer Types

**Files**: Any remaining layer files (`Sigmoid.hpp`, `Tanh.hpp`, `Softmax.hpp`, etc.)

**General Pattern**:

1. Replace `Eigen::MatrixXf` member variables with `nn::Tensor`
2. Update forward methods to use Tensor operations
3. Update backward methods to accumulate gradients properly
4. Add any missing element-wise operations to Tensor class

### Step 5.5: Add Missing Tensor Operations

Based on layer migration needs, add these methods to `nn::Tensor`:

**Matrix Operations**:

```cpp
Tensor matmul(const Tensor& other) const;  // Matrix multiplication
Tensor transpose() const;                   // Matrix transpose
```

**Element-wise Operations**:

```cpp
Tensor relu() const;                        // max(0, x)
Tensor sigmoid() const;                     // 1 / (1 + exp(-x))
Tensor tanh() const;                        // tanh(x)
Tensor softmax() const;                     // softmax along last dimension
```

**Reduction Operations**:

```cpp
Tensor sum(int dim = -1) const;            // Sum along dimension
Tensor mean(int dim = -1) const;            // Mean along dimension
float sum_all() const;                      // Sum all elements
```

**Random Operations**:

```cpp
void random_normal(float mean = 0.0f, float std = 1.0f);  // Normal distribution
void random_uniform(float min = 0.0f, float max = 1.0f);  // Uniform distribution
```

### Step 5.6: Update Layer Interface

**File**: `src/core/layers/Module.h`

Consider updating the `Module` interface to work exclusively with `nn::Tensor`:

```cpp
// Current
virtual Tensor forward(const Tensor& input) = 0;
virtual Tensor backward(const Tensor& grad_output) = 0;

// Ensure all implementations use Tensor consistently
```

### Step 5.7: Migrate Wave Processing Files

**Files**: `src/core/wave/*.cpp`

**Current Issues**:

- May use Eigen for FFT operations or signal processing

**Migration Steps**:

1. Replace any `Eigen::VectorXf` with `nn::Tensor`
2. Update function signatures
3. Ensure compatibility with FFTW3 integration

### Step 5.8: Clean Up Eigen Includes

**Global Cleanup**:

1. **Remove unnecessary Eigen includes**:
   - Remove `#include <Eigen/Dense>` from all migrated files
   - Keep only in `Tensor.hpp/cpp` and files that still need direct Eigen access

2. **Update CMakeLists.txt**:
   - Ensure Eigen is still available for Tensor's internal implementation
   - Remove Eigen dependencies from targets that no longer need them

3. **Update header guards and dependencies**:
   - Ensure forward declarations work correctly
   - Update any conditional compilation

### Step 5.9: Performance Optimization

**After Migration**:

1. **Profile performance**:

   ```bash
   # Use perf or valgrind to measure performance impact
   perf record ./bin/experiment_02
   perf report
   ```

2. **Optimize Tensor operations**:
   - Ensure Eigen's optimizations are still effective through Tensor
   - Consider adding `noalias()` equivalents in Tensor methods
   - Profile memory usage and cache efficiency

3. **Add performance regression tests**:
   - Benchmark key operations before/after migration
   - Set performance thresholds in CI

### Step 5.10: Final Integration Testing

**Comprehensive Testing**:

1. **Run all experiments**:

   ```bash
   cd /home/ensismoebius/Repos/doutorado/software/nn/build
   ctest --output-on-failure -j$(nproc)
   ```

2. **Test layer functionality**:

   ```bash
   # Test individual layers
   ./bin/test_layers

   # Test full networks
   ./bin/test_networks
   ```

3. **Memory leak testing**:

   ```bash
   valgrind --leak-check=full ./bin/experiment_02
   ```

4. **Cross-platform testing**:
   - Test on different architectures
   - Verify numerical stability across platforms

### Step 5.11: Documentation Updates

**Update Documentation**:

1. **Update copilot-instructions.md**:
   - Document new Tensor methods
   - Update API contracts
   - Add migration notes

2. **Update CHANGELOG.md**:
   - Document breaking changes
   - List new features
   - Note performance impacts

3. **Update README.md**:
   - Document Tensor as the primary abstraction
   - Update build instructions if changed

### Step 5.12: Deprecation and Compatibility

**Backward Compatibility**:

1. **Add deprecation warnings** for any remaining Eigen usage
2. **Provide migration utilities** if needed
3. **Document upgrade path** for external users

### Step 5.13: Future Backend Flexibility

**Prepare for Multiple Backends**:

1. **Abstract Tensor Interface**:

   ```cpp
   class TensorBackend {
   public:
       virtual ~TensorBackend() = default;
       virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;
       virtual Tensor relu(const Tensor& x) = 0;
       // ... other operations
   };
   ```

2. **Backend Registry**:
   - Allow switching between Eigen, PyTorch, custom implementations
   - Configuration-based backend selection

## Phase 5 Success Criteria

- ✅ All layer files use `nn::Tensor` exclusively
- ✅ No direct Eigen usage in application code
- ✅ All tests pass with no performance regression >5%
- ✅ Memory usage remains stable
- ✅ Numerical results identical to Eigen-based implementation
- ✅ Clean separation between Tensor API and Eigen implementation

## Phase 5 Timeline

- **Week 1-2**: Migrate Linear and simple activation layers
- **Week 3-4**: Migrate Conv2d layer (most complex)
- **Week 5**: Migrate remaining layers and wave processing
- **Week 6**: Performance testing and optimization
- **Week 7**: Documentation and final integration testing

## Phase 5 Risk Mitigation

1. **Incremental Migration**: Migrate one layer at a time with full testing
2. **Performance Monitoring**: Stop if performance regression exceeds threshold
3. **Fallback Plan**: Keep Eigen-based backup branches
4. **Numerical Validation**: Compare outputs between Eigen and Tensor versions

---

## Common Patterns to Replace

| Eigen Code                      | Tensor Equivalent                 |
| ------------------------------- | --------------------------------- |
| `Eigen::MatrixXf m(rows, cols)` | `nn::Tensor m(rows, cols)`        |
| `m.row(i)`                      | `m.row(i)` (add method)           |
| `m.col(j)`                      | `m.col(j)` (add method)           |
| `m.block(r,c,hr,wc)`            | `m.block(r,c,hr,wc)` (add method) |
| `m.array() += val`              | `m.add_scalar(val)` (add method)  |
| `Eigen::MatrixXf::Zero(r,c)`    | `nn::Tensor(r,c)` (default zero)  |

## Potential Issues

1. **Performance**: Direct Eigen access may be faster; measure impact
2. **Int Types**: Confusion matrix uses int; may need separate IntTensor
3. **Complex Operations**: Some Eigen operations may be hard to abstract
4. **Third-party Integration**: External libraries expecting Eigen matrices

## Rollback Plan

- Keep Eigen includes commented out rather than deleted
- Use git branches for phased migration: `feature/tensor-migration-phase1`, etc.
- Have backup of working Eigen-based code
- Document all changes in commit messages

## Version Control Strategy

- **Branching**: Create feature branches for each phase
- **Commits**: Make atomic commits for each file change
- **Testing**: Run full test suite after each major change
- **Merging**: Only merge after all tests pass and integration works

## Future Considerations

- Backend Interface: Define abstract tensor interface for different backends
- GPU Support: Ensure new backend can support GPU operations
- Serialization: Update save/load methods for Tensor

## Timeline

- **Week 1**: Implement missing Tensor methods
- **Week 2**: Migrate high-priority files
- **Week 3**: Migrate medium-priority files
- **Week 4**: Testing, bug fixes, and validation

---

_This migration is a significant undertaking. Proceed incrementally and test thoroughly at each step._
