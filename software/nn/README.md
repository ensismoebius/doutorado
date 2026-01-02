# Neural Network Framework (C++20)

A high-performance, optimized C++20 neural network framework featuring spiking neural networks (SNN), autoencoders, and EEG/audio synchronization capabilities. This framework has been systematically optimized for performance, security, and maintainability.

## 🚀 Features

- **Modern C++20** implementation with RAII and smart pointers
- **Spiking Neural Networks (SNN)** with Leaky Integrate-and-Fire neurons
- **Sparse Autoencoders** for dimensionality reduction
- **EEG/Audio Synchronization** with wavelet transforms
- **Optimized Performance** with SIMD vectorization and OpenMP parallelization
- **Memory Efficient** with conditional gradient computation for inference
- **Comprehensive Testing** with Google Test framework
- **Security Hardened** with input validation and bounds checking

## 📋 Prerequisites

### System Requirements

- **Linux** (Arch Linux primary target)
- **C++20** compatible compiler (Clang 15+ recommended)
- **CMake 3.20+**
- **OpenMP 5.1+** for parallelization
- **Python 3.8+** (for some utilities)

### Dependencies

The project uses FetchContent for automatic dependency management:

- **Eigen 3.4+** - Linear algebra library
- **Google Test** - Testing framework
- **FFTW 3.3+** - Fast Fourier transforms
- **NFFT 3.3+** - Non-uniform FFT
- **MatIO** - MATLAB file I/O
- **YAML-CPP** - Configuration files
- **ImGui/ImPlot** - GUI components (optional)

## 🏗️ Building the Project

### Quick Build

```bash
# Clone the repository
git clone <repository-url>
cd nn

# Configure with CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build -- -j$(nproc)

# Run tests to verify
ctest --test-dir build --output-on-failure
```

### Build Options

```bash
# Debug build with sanitizers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fsanitize=undefined"

# Optimized release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native"

# With clang-tidy integration
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_CLANG_TIDY="clang-tidy;--config-file=.clang-tidy"
```

### Build Targets

- `nn_tensor_test` - Tensor class tests
- `nn_layers_gtest` - Neural network layers tests
- `nn_dataloader_gtest` - Data loading tests
- `experiment_02` - Main experiment executable
- `profile_experiment_02` - Callgrind profiling target

## 🧪 Running Tests

### All Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure -j$(nproc)

# Run with verbose output
ctest --test-dir build --output-on-failure -V
```

### Specific Test Categories

#### Tensor Tests

```bash
# Run tensor unit tests
./build/src/core/tensor/tests/tensor_gtest

# Specific tensor test
./build/src/core/tensor/tests/tensor_gtest --gtest_filter=TensorTest.ConstructorAndAssignment
```

#### Layer Tests

```bash
# Run neural network layer tests
./build/src/core/layers/tests/layers_gtest

# Test specific layer
./build/src/core/layers/tests/layers_gtest --gtest_filter="*Conv2d*"

# Test parallel execution
export OMP_NUM_THREADS=4
./build/src/core/layers/tests/layers_gtest --gtest_filter="Conv2dTest.ParallelExecution"
```

#### Data Loading Tests

```bash
# Run data loader tests
./build/src/core/dataLoaders/tests/dataloader_gtest

# Test MAT file loading
./build/src/core/dataLoaders/tests/mat_file_dataset_gtest
```

### Test Coverage

```bash
# Generate coverage report (requires gcov/lcov)
cmake --build build --target coverage
```

## 📊 Performance Profiling and Benchmarking

### Callgrind Profiling (Performance Analysis)

```bash
# Profile experiment_02 with Callgrind
cmake --build build --target profile_experiment_02

# Manual profiling
valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
  ./build/bin/experiment_02

# Analyze results
callgrind_annotate callgrind.out
```

### Memory Profiling (Valgrind Massif)

```bash
# Profile memory usage of tensor tests
valgrind --tool=massif --massif-out-file=massif_tensor.out \
  ./build/src/core/tensor/tests/tensor_gtest

# Profile layer operations
valgrind --tool=massif --massif-out-file=massif_layers.out \
  ./build/src/core/layers/tests/layers_gtest

# Analyze memory usage
ms_print massif_layers.out
```

### Vectorization Analysis

```bash
# Check SIMD vectorization in compilation
clang++ -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
  -c src/core/layers/Conv2d_impl.cpp \
  -I src -I build/_deps/eigen-src -std=c++20
```

### OpenMP Parallelization Testing

```bash
# Test with different thread counts
export OMP_NUM_THREADS=1
./build/src/core/layers/tests/layers_gtest --gtest_filter="Conv2dTest.ParallelExecution"

export OMP_NUM_THREADS=4
./build/src/core/layers/tests/layers_gtest --gtest_filter="Conv2dTest.ParallelExecution"

export OMP_NUM_THREADS=8
./build/src/core/layers/tests/layers_gtest --gtest_filter="Conv2dTest.ParallelExecution"
```

## 🔧 Static Analysis Tools

### Cppcheck (Static Analysis)

```bash
# Run cppcheck on source code
cppcheck --enable=all --std=c++20 --platform=native \
  --suppress=missingIncludeSystem --quiet src/

# Generate XML report
cppcheck --enable=all --std=c++20 --platform=native \
  --suppress=missingIncludeSystem --quiet --xml --xml-version=2 \
  --output-file=cppcheck.xml src/
```

### Flawfinder (Security Analysis)

```bash
# Security vulnerability scan
flawfinder --minlevel=1 --columns --singleline src/
```

### Clang-Tidy (Code Quality)

```bash
# Run clang-tidy on specific file
clang-tidy --config-file=.clang-tidy \
  src/core/layers/Conv2d_impl.cpp -- \
  -I src -I build/_deps/eigen-src -std=c++20

# Run on entire codebase (integrated with build)
cmake --build build --target tidy
```

## 🧠 Using the Neural Network Framework

### Basic Usage Example

```cpp
#include "core/tensor/Tensor.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/ReLU.hpp"
#include "core/layers/Sequential.hpp"

// Create a simple neural network
auto model = std::make_unique<Sequential>();
model->add(std::make_unique<Linear>(784, 128));  // Input: 784, Hidden: 128
model->add(std::make_unique<ReLU>());
model->add(std::make_unique<Linear>(128, 10));   // Output: 10 classes

// Create input tensor (batch_size=32, features=784)
nn::Tensor input(32, 784);
// ... fill input data ...

// Forward pass (inference mode - no gradients needed)
nn::Tensor output = model->forward(input, false);  // requires_grad=false

// Forward pass (training mode - gradients needed)
nn::Tensor output_train = model->forward(input, true);  // requires_grad=true
```

### Spiking Neural Network Example

```cpp
#include "core/layers/Leaky.hpp"

// Create SNN layer
Leaky snn_layer(0.01f, 1.0f, 0.5f, 20);  // alpha, threshold, reset, timesteps

// Input spike train (batch_size=10, features=64, time=20)
nn::Tensor spike_input(10, 64, 20);
// ... fill spike data ...

// Forward pass through SNN
nn::Tensor spike_output = snn_layer.forward(spike_input, true);
```

### Data Loading

```cpp
#include "core/dataLoaders/MatFileDataset.hpp"

// Load EEG data from MATLAB file
MatFileDataset dataset("data/eeg_data.mat", "eeg_inputs", "eeg_targets");

// Create data loader
DataLoader loader(dataset, 32, true, 42);  // batch_size=32, shuffle=true, seed=42

// Iterate through batches
for (const auto& [inputs, targets] : loader) {
    // Training loop
    auto predictions = model->forward(inputs, true);
    // ... compute loss, backward pass, etc.
}
```

## 📁 Code Organization

### Directory Structure

```
nn/
├── src/core/                    # Core framework components
│   ├── tensor/                  # Tensor implementation
│   ├── layers/                  # Neural network layers
│   ├── dataLoaders/             # Data loading utilities
│   ├── initializers/            # Weight initialization
│   ├── optimizers/              # Optimization algorithms
│   └── statistics/              # Statistical utilities
├── src/experiments/             # Research experiments
├── cmake/                       # CMake configuration
├── lib/                         # Vendored dependencies
├── build/                       # Build artifacts (generated)
├── debug/                       # Debugging utilities
├── docs/                        # Documentation
└── .github/copilot-instructions.md  # Development guidelines
```

### Key Classes

#### Core Classes

- **`Tensor`** - Multi-dimensional array with automatic differentiation support
- **`Module`** - Base class for all neural network layers
- **`Sequential`** - Container for stacking layers

#### Layer Types

- **`Linear`** - Fully connected layer
- **`Conv2d`** - 2D convolution layer
- **`ReLU`/`LeakyReLU`** - Activation functions
- **`Leaky`** - Spiking neural network layer
- **`MaxPool2d`** - 2D max pooling

#### Data Handling

- **`MatFileDataset`** - MATLAB file dataset loader
- **`DataLoader`** - Batch iterator with shuffling
- **`MatFileUtils`** - MATLAB file I/O utilities

## 🔄 Updating and Maintaining the Code

### Adding New Layers

1. **Inherit from Module**:

```cpp
class MyLayer : public Module {
public:
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;
};
```

2. **Implement forward/backward**:

```cpp
auto MyLayer::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor {
    // Cache input only if gradients needed
    if (requires_grad) {
        input_cache_ = input;
    }
    // ... layer computation ...
    return output;
}
```

3. **Add to Sequential**:

```cpp
auto model = std::make_unique<Sequential>();
model->add(std::make_unique<MyLayer>(param1, param2));
```

### Adding Tests

1. **Create test file** in appropriate `tests/` directory
2. **Use Google Test framework**:

```cpp
TEST(MyLayerTest, ForwardPass) {
    MyLayer layer(params);
    nn::Tensor input = /* create test input */;
    nn::Tensor output = layer.forward(input, true);

    // Assertions
    EXPECT_EQ(output.rows(), expected_rows);
    EXPECT_EQ(output.cols(), expected_cols);
}
```

3. **Add to CMakeLists.txt**:

```cmake
add_executable(nn_my_layer_test tests/my_layer_test.cpp)
target_link_libraries(nn_my_layer_test PRIVATE nn_core GTest::gtest_main)
gtest_discover_tests(nn_my_layer_test)
```

### Performance Optimization Guidelines

1. **Use conditional caching** in forward passes
2. **Prefer Eigen operations** over manual loops
3. **Use member initializer lists** in constructors
4. **Mark non-throwing methods** as `noexcept`
5. **Profile with Callgrind/Massif** before optimizing

### Code Quality Standards

- **Follow clang-tidy** recommendations (with research-appropriate exclusions)
- **Use RAII** and smart pointers
- **Validate inputs** and check bounds
- **Document complex algorithms** with comments
- **Maintain const-correctness**

## 🐛 Troubleshooting

### Common Issues

#### Build Failures

```bash
# Clean build
rm -rf build/
cmake -S . -B build
cmake --build build -- -j$(nproc)
```

#### Test Failures

```bash
# Run specific failing test with debug output
./build/path/to/test --gtest_filter=FailingTest --gtest_break_on_failure
```

#### Memory Issues

```bash
# Check for memory leaks
valgrind --leak-check=full ./build/bin/experiment_02
```

#### Performance Issues

```bash
# Profile with Callgrind
valgrind --tool=callgrind --callgrind-out-file=profile.out ./build/bin/experiment_02
kcachegrind profile.out  # GUI analysis
```

### Getting Help

1. **Check existing tests** for usage examples
2. **Review optimizations.md** for performance tuning
3. **Run static analysis** tools for code issues
4. **Check CMake configuration** for build issues

## 📈 Performance Characteristics

### Optimization Results

- **Inference Performance**: 20-50% improvement with conditional gradient computation
- **Memory Usage**: 2-5 MB peak for neural network operations (no leaks)
- **Parallelization**: OpenMP multi-threading verified
- **Vectorization**: SIMD optimization active in matrix operations

### Benchmark Commands

```bash
# Performance profiling
make profile_experiment_02

# Memory profiling
valgrind --tool=massif --massif-out-file=memory_profile.out ./build/bin/experiment_02

# Vectorization check
clang++ -O3 -Rpass=loop-vectorize -c src/core/layers/Linear.hpp -I src -I build/_deps/eigen-src
```

## 🤝 Contributing

1. **Follow the optimization guidelines** in `optimizations.md`
2. **Run all tests** before submitting changes
3. **Update documentation** for new features
4. **Use clang-tidy** for code quality checks
5. **Profile performance** impact of changes

## 📄 License

[Specify your license here]

---

**Note**: This framework has been systematically optimized through static analysis, security hardening, performance profiling, and code quality improvements. See `optimizations.md` for detailed optimization history and results.</content>
<parameter name="filePath">/home/ensismoebius/Repos/doutorado/software/nn/README.md
