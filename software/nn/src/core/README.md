# Core Library Overview

This directory contains the project's core C++ components: data loaders, tensor operations, layers, optimizers, utilities, and related helpers used by experiments and demos.

Structure
- `data_loaders/` – Dataset and DB-backed batch sources (mat, sqlite, windowing).
- `tensor/` – Tensor and linear algebra abstractions.
- `layers/` – Neural network layers and layer utilities.
- `optimizers/` – Optimizer implementations (Adam, SGD helpers).
- `saver/` & `serialization/` – Model/optimizer state save/load helpers.
- `utility/`, `tools/` – Misc utilities and helper programs.

Usage
- Include public headers from `include/nn/...` (e.g. `include/nn/tensor/...`, `include/nn/layers/...`).
- Link against the appropriate CMake target (matching the folder name, e.g. `data_loaders`, `layers`, `optimizers`).

Building and Tests
- The project uses CMake. From the repo root:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
  cmake --build build -j$(nproc)
  ctest --test-dir build --output-on-failure -j4
  ```

Where to look
- For documentation and usage examples, see the `src/core/*/tests/` GTest files which illustrate typical API usage.
