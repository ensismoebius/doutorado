# CMake Build System

Documents the `nn` project CMake configuration, module layout, and usage workflows.

---

## Project Structure

The root `CMakeLists.txt` is intentionally thin — it delegates all logic to modular `cmake/*.cmake` includes, then adds three source subtrees:

```
CMakeLists.txt          ← project declaration, include order, add_subdirectory
cmake/                  ← modular CMake helpers (see below)
src/
  core/                 ← reusable library modules (tensor, layers, optimizers, …)
  demos/                ← standalone demo executables
  experiments/          ← numbered experiment executables + libs (00–04)
include/nn/             ← public headers for all core modules
```

### CMake Module Responsibilities

| Module | Role |
|--------|------|
| `cmake/Policies.cmake` | Sets CMake compatibility policies (e.g. CMP0069) |
| `cmake/Tooling.cmake` | ccache detection, `compile_commands.json` export, analysis tool detection |
| `cmake/DevAndAnalysisTargets.cmake` | Custom targets: `dev-setup`, `analysis-all`, `clean-cache`, `check_xtensor_leaks` |
| `cmake/Flags.cmake` | C++20 standard, PIC, linker selection (mold → lld fallback), CMake options |
| `cmake/EnableCoverage.cmake` | `--coverage` flags when `NN_ENABLE_COVERAGE=ON` |
| `cmake/PrecompiledHeaders.cmake` | Per-target PCH helpers (opt-in via `NN_ENABLE_PCH`) |
| `cmake/PackageChecking.cmake` | System deps: xtensor, OpenMP, SDL2, BLAS/LAPACK/OpenBLAS, OpenCL |
| `cmake/VendorIncludes.cmake` | Aggregates all vendored deps — includes each `Vendor*.cmake` in order |
| `cmake/SanitizerFlags.cmake` | ASan + UBSan for Debug/RelWithDebInfo (opt-in via `NN_ENABLE_ASAN=ON`) |

### Vendored Dependencies (via `cmake/VendorIncludes.cmake`)

| Vendor module | Library |
|---------------|---------|
| `VendorCnpy.cmake` | cnpy (NumPy .npy/.npz I/O) |
| `VendorArgparse.cmake` | argparse |
| `VendorCLI11.cmake` | CLI11 |
| `VendorFFTW.cmake` | FFTW3 |
| `VendorNFFT3.cmake` | NFFT3 |
| `VendorImgui.cmake` | Dear ImGui |
| `VendorImplot.cmake` | ImPlot |
| `VendorGTest.cmake` | GoogleTest (`GTest::GTest`, `GTest::Main`) |
| `VendorMatio.cmake` | matio (`MATIO::MATIO`) |
| `VendorMatioCppShim.cmake` | matio-cpp shims/workarounds |
| `VendorMatplotlibCpp.cmake` | matplotlib-cpp |
| `VendorYaml.cmake` | yaml-cpp |
| `VendorJson.cmake` | nlohmann::json |
| `VendorXtensorParallel.cmake` | xtensor parallelization settings |
| `VendorSqlite.cmake` | Optional vendored SQLite amalgamation |

### System Dependencies (via `cmake/PackageChecking.cmake`)

Required at configure time:

- **xtensor** — linear algebra
- **OpenMP** — parallel loops
- **SDL2** — GUI/audio demos
- **BLAS / LAPACK / OpenBLAS** — linear algebra acceleration
- **OpenCL** *(optional)* — GPU tensor backend; warning emitted if absent

---

## CMake Options

| Option | Default | Effect |
|--------|---------|--------|
| `NN_ENABLE_FAST_LINKER` | `ON` | Auto-selects mold → lld for faster linking |
| `NN_USE_OBJECT_LIBRARIES` | `ON` | OBJECT libs on select targets to cut relink overhead |
| `NN_ENABLE_PCH` | `ON` | Per-target precompiled headers |
| `NN_ENABLE_ASAN` | `OFF` | AddressSanitizer + UBSan on Debug/RelWithDebInfo |
| `NN_ENABLE_COVERAGE` | `OFF` | `--coverage` instrumentation for gcov/lcov reports |

---

## Presets Overview

Defined in `CMakePresets.json`. Build outputs land in `out/build/<preset>/`.

| Preset | Build Type | Key Flags | Use Case |
|--------|-----------|-----------|----------|
| `Clang_20.1.8_x86_64-pc-linux-gnu` | Debug | `-g -O0`, Clang 20 | Development, debugging |
| `max-performance` | Release | `-O3 -march=native -flto`, ccache, Ninja | Production / benchmarking |
| `pgo-generate` | Release | `-O3 -march=native -fprofile-generate`, ccache | PGO instrumented build |
| `pgo-use` | Release | `-O3 -march=native -fprofile-use -fprofile-correction -flto`, ccache | Final PGO-optimized build |

All Release presets use Ninja. The `max-performance` and `pgo-use` presets set `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` (LTO). OpenMP is explicitly configured in `max-performance` to avoid auto-detection failures.

---

## Quick Start

```bash
# Configure + build (recommended default)
cmake --preset=max-performance
cmake --build --preset=max-performance -j$(nproc)

# Run all tests
ctest --preset=max-performance --output-on-failure -j4
```

For a plain out-of-tree Debug build without presets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -j4
```

---

## Common Workflows

### Development Cycle

```bash
# Edit source
vim src/core/tensor/Tensor.cpp

# Incremental build (Ninja, fast)
cmake --build --preset=max-performance -j$(nproc)

# Run a targeted test
ctest --test-dir out/build/max-performance -R tensor_gtest
```

### Full Rebuild

```bash
rm -rf out/build/max-performance
cmake --preset=max-performance
cmake --build --preset=max-performance -j$(nproc)
```

### Enable AddressSanitizer

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DNN_ENABLE_ASAN=ON
cmake --build build-asan -j$(nproc)
ctest --test-dir build-asan --output-on-failure -j1
```

### Enable Coverage

```bash
cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DNN_ENABLE_COVERAGE=ON
cmake --build build-coverage -j$(nproc)
ctest --test-dir build-coverage --output-on-failure -j4
# Collect coverage with lcov (use --ignore-errors inconsistent for gtest macros)
lcov --capture --directory build-coverage --output-file coverage.info \
     --ignore-errors inconsistent
lcov --remove coverage.info '/usr/*' '*/_deps/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage-html
```

### Profile-Guided Optimization (PGO)

```bash
# 1. Instrumented build
cmake --preset=pgo-generate
cmake --build --preset=pgo-generate -j$(nproc)

# 2. Run a representative workload to collect profiles
./out/build/pgo-generate/src/experiments/03/experiment03 \
    --profile src/experiments/03/profiles/sample.json

# 3. Final optimized build using collected profiles
cmake --preset=pgo-use
cmake --build --preset=pgo-use -j$(nproc)
```

See [PGO.md](./PGO.md) for full guidance.

---

## Source Subtree Layout

### `src/core/` — Core Library Modules

Each module under `src/core/` has its own `CMakeLists.txt` and optional `tests/` sub-directory.

| Module | CMake target(s) |
|--------|----------------|
| `tensor/` | `tensor` |
| `layers/` | `layers` |
| `optimizers/` | `optimizers` |
| `dataLoaders/` | `dataLoaders`, samplers, 10.1117 dataset |
| `initializers/` | `initializers` |
| `linearAlgebra/` | `linearAlgebra` |
| `models/autoencoder/` | autoencoder model |
| `models/lstm/` | LSTM model |
| `statistics/` | `statistics` |
| `wave/` | `wave` |
| `wavelet/` | `wavelet` |
| `paraconsistent/` | `paraconsistent` |
| `saver/` | `saver` |
| `utility/` | `utility` |
| `tools/` | internal tools |
| `training/` | `training` |

All core modules are included from `src/core/CMakeLists.txt` via `add_subdirectory`.  
Sanitizer flags (`SanitizerFlags.cmake`) are applied at the `src/core/` level.

### `src/experiments/` — Numbered Experiments

Shared config library `experiments_config` (JSON-backed) is built at the `src/experiments/` level and linked by each numbered experiment.

| Experiment | Directory |
|-----------|-----------|
| 00 | `src/experiments/00/` |
| 01 | `src/experiments/01/` |
| 02 | `src/experiments/02/` |
| 03 | `src/experiments/03/` (Autoencoder / SNN) |
| 04 | `src/experiments/04/` (LSTM comparative) |

Each experiment may contain a `lib/` subdirectory for reusable components and a `tests/` subdirectory with GTest targets.

### `src/demos/` — Standalone Demos

| Demo | Directory |
|------|-----------|
| C++ demos | `src/demos/cppdemos/` |
| Spiking network plot | `src/demos/exec_plotSpikingNetwork/` |
| ResNet demo | `src/demos/exec_resnet_demo/` |
| FFTW3 demo | `src/demos/fftw3_demo/` |
| LFCC pipeline | `src/demos/lfcc_pipeline/` |
| Voice biometrics | `src/demos/voice_biometrics_cpp/` |
| Wavelet demo | `src/demos/wavelet_demo/` |

---

## Running Tests

```bash
# All tests
ctest --preset=max-performance --output-on-failure -j4

# Specific test suite by name pattern
ctest --preset=max-performance -R dataLoaders_gtest
ctest --preset=max-performance -R tensor_gtest
ctest --preset=max-performance -R optimizers_gtest

# Single test with verbose output
ctest --preset=max-performance -R tensor_gtest --output-on-failure -j1
```

Test binaries land next to their `CMakeLists.txt` source in `out/build/<preset>/src/...`.

---

## Development Targets

Available as CMake custom targets (defined in `cmake/DevAndAnalysisTargets.cmake`):

```bash
# Check for required developer tools
cmake --build out/build/max-performance --target dev-setup

# Run all static analysis checks
cmake --build out/build/max-performance --target analysis-all

# Clear ccache
cmake --build out/build/max-performance --target clean-cache
```

---

## Environment Variables

| Variable | Purpose | Recommended value |
|----------|---------|-------------------|
| `OMP_NUM_THREADS` | OpenMP thread count | `1` (for reproducible experiments) |
| `OPENBLAS_NUM_THREADS` | OpenBLAS threads | `1` |
| `MKL_NUM_THREADS` | MKL threads | `1` |
| `CC` / `CXX` | Override compiler | `clang` / `clang++` |

---

## Troubleshooting

### CMake version too old
```bash
cmake --version  # Requires ≥ 3.10 (3.28+ recommended for full preset support)
```

### Compiler not found / wrong compiler
```bash
# Use clang explicitly
env CC=clang CXX=clang++ cmake -S . -B build
```

### ccache not accelerating
```bash
ccache --version   # Confirm installed
ccache -s          # Check hit rate
```

### Slow incremental links
- `NN_ENABLE_FAST_LINKER=ON` (default) auto-selects mold then lld.
- Confirm with: `cmake --build ... -- -v 2>&1 | grep fuse-ld`

### Test failures under sanitizers
```bash
# Run single-threaded for cleaner output
ctest --test-dir build-asan --output-on-failure -j1
```

### lcov "inconsistent" errors (coverage builds)
Use `--ignore-errors inconsistent` — caused by gtest `TEST()` macro DWARF line mismatch.  
See [GRID_TESTS_COMPREHENSIVE_RUNBOOK](./Grid-Runbook.md) for the full coverage workflow.
(lldb) run --gtest_filter=TensorGtest.test_name

## See Also

- [Architecture](./Architecture.md) - System design  
- [Static Analysis](./Guides/Static-Analysis.md) - Code quality  
- [PGO](./Guides/PGO.md) - Optimization workflow