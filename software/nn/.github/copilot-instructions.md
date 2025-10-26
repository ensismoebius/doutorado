# Neural Network Framework Guidelines (updated)

This C++ neural network framework focuses on spiking neural networks (SNNs) and small autoencoder examples. The project uses Eigen for linear algebra, GoogleTest for unit tests, and CMake for builds. This document outlines up-to-date patterns, conventions and where to find common components in the repository.

## Project layout (high level)

```
nn/
├── src/
│   ├── dataLoaders/      # Dataset, DataLoader, MatFile, tests
│   ├── initializers/     # Weight initialization strategies
│   ├── layers/           # Neural network layers (Linear, Leaky, ...)
│   ├── optimizers/       # Optimization algorithms (Adam, SGD, ...)
│   ├── tensor/           # Tensor wrapper around Eigen
│   └── util/             # Utility functions and helpers
├── lib/                  # Third-party libs (cnpy, imgui, implot, ...)
├── .github/              # repo guidelines & Copilot instructions
├── build/                # CMake build artifacts (created at configure-time)
├── Flags.cmake           # Compiler/flag policy used by top-level CMake
├── Main.cmake            # High-level project wiring (includes modular .cmake files)
└── README_ACTION_HISTORY.md (generated)  # optional auto-generated action history
```

## Key components and current conventions

### Tensor system

- `Tensor` is implemented in `src/tensor/Tensor.hpp` and wraps Eigen matrices.
- A Tensor contains `data` (Eigen::MatrixXf) and `grad` where applicable.

### Layers and Modules

- Layers inherit from a `Module` base and implement `forward()` and `backward()`.
- Common layers: `Linear`, `Leaky` (LIF-like), `ReLU`, `LeakyReLU` etc.
- Composition: use `Sequential` (`src/layers/Sequential.hpp`) to chain layers.

### Data loading

- Dataset API: abstract `Dataset` in `src/dataLoaders/Dataset.h` with `get_item`, `collate`, `size()`.
- `TensorDataset` in `src/dataLoaders/TensorDataset.h` is a concrete dataset that wraps `Tensor` inputs/targets and implements `get_item` and `collate`.
- `DataLoader` lives in `src/dataLoaders/DataLoader.h/.cpp` and provides iteration, optional shuffling, deterministic seeding, and batching. Use `for (const auto &batch : loader)` to iterate.
- `MatFile` utilities: `src/dataLoaders/MatFile.cpp/h` read/write MATLAB `.mat` v5 files and are used by the demo.

Notes on DataLoader/Dataset
- The code was refactored to be PyTorch-like: `Dataset` + `DataLoader` + `collate` semantics.
- `TensorDataset` uses `Tensor::slice()` to build per-sample batches. `Dataset::collate` provides a default implementation that allocates and fills matrices.

### Demo: `dataLoader_demo`

- Demo source: `src/dataLoaderTest.cpp`.
- Behavior: reads the first numeric variable from a `.mat` file (via `MatFile`), trims to safe dimensions, wraps data into `TensorDataset`, constructs a tiny autoencoder (Linear -> Leaky -> Linear), trains with `MSELoss` + `Adam`, and writes `reconstructed.mat`.
- Defensive features: sample-size caps (`max_features`, `max_elements`) and try/catch around large Eigen allocations to avoid OOMs when users point to large MAT files.

### Initializers & optimizers

- Initializers: `xavierInitializer`, `kaimingSNNInitializer` in `src/initializers/`.
- Optimizers: `Adam`, `SGD`, `SGDMinimal` implementing `attach(params)`, `step()`, `zero_grad()`.

## Build & CMake layout

- Top-level CMakeLists includes `Flags.cmake` to set compiler flags and then includes `Main.cmake`.
- I reorganized `Main.cmake` to include modular files so each logical group lives in its own `.cmake` file at the repository root, e.g.:
   - `AutoEncoderTargets.cmake` — auto-encoder related targets
   - `PlotTarget.cmake` — plotSpikingNetwork and ImGui/ImPlot wiring
   - `LoadingDataTarget.cmake` — `loadingData` target
- `Flags.cmake` contains compiler flags and policy settings (C++20, diagnostics, debug/release flags) and is included by the top-level `CMakeLists.txt`.

Quick configure & build

```bash
cmake -S . -B build
cmake --build build -- -j <cpus>
```

## Tests

- Unit tests: GoogleTest (gtest). Tests are grouped and discovered via `gtest_discover_tests`.
- Consolidated test target: `dataLoaders_gtest` (covers MatFile tests and data loader tests). Other component gtest targets (layers_gtest, optimizers_gtest, tensor_gtest, etc.) are found under `src/*/CMakeLists.txt`.
- Run tests (from repo root):

```bash
cmake --build build --target dataLoaders_gtest
ctest --test-dir build --output-on-failure -j 4
```

## Tests added recently (high level)
- `src/dataLoaders/dataLoader_gtest.cpp` — deterministic shuffle, small/empty dataset behavior.
- `src/dataLoaders/dataLoader_more_gtest.cpp` — collate correctness, mismatched input/target columns, deterministic seed checks, and a lightweight concurrency smoke test.
- CMake updated to create a single `dataLoaders_gtest` target that links the loader implementation so tests link cleanly.

## Conventions & coding guidelines (current)

1. Code style

    - Use `auto` with trailing-return type when helpful for clarity.
    - Member variables use snake_case; functions use camelCase.
    - Prefer descriptive local variable names in demos and tests.

2. Memory & ownership

    - Use `std::shared_ptr` for layer ownership and dataset sharing.
    - Tensor slices/copied views are used for safety; avoid exposing raw pointers for internal tensors.

3. Error handling

    - Use `assert` for internal invariants.
    - For I/O and external input (MAT files) use explicit checks and guarded allocations.

4. Performance

    - Use Eigen efficiently (avoid unnecessary copies where possible).
    - Configure `Eigen` parallel behavior via `EigenParallel.cmake` and `configure_eigen_parallel_target()`.
    - Batch-level parallelism uses OpenMP; targets are linked with `${OpenMP_CXX_LIBRARIES}`.

## Common tasks (examples)

1) Add a new layer

```cpp
struct NewLayer : public Module {
   auto forward(const Tensor &input) -> Tensor override;
   auto backward(const Tensor &grad_output) -> Tensor override;
};
```

2) Small training loop sketch

```cpp
optimizer.zero_grad(params);
Tensor out = model.forward(input);
Tensor loss = loss_fn.forward(out);
Tensor grad = loss_fn.backward(out);
model.backward(grad);
optimizer.step(params);
```

## Helpful pointers

- `src/dataLoaders/TensorDataset.h` — example of a minimal concrete Dataset.
- `src/dataLoaders/DataLoader.{h,cpp}` — iteration/shuffle/batching behavior (iterator-based API).
- `src/dataLoaderTest.cpp` — a runnable demo that trains an autoencoder on a MATLAB variable and writes `reconstructed.mat`.
- `Flags.cmake` — common compiler flags and standard settings used across the project.

## Next steps and optional improvements

- Add a committed small `.mat` test file for CI reproducibility (currently demo creates `/tmp/dataLoader_demo_default.mat` at runtime).
- Add sanitizer-enabled CMake configurations (`ASAN`, `TSAN`) and a CI job to run concurrency-related tests under ThreadSanitizer.
- Split `cmake/` helpers into a small helper directory (`cmake/utils.cmake`) and document available helper functions.

If you want, I can:
- generate a `README_ACTION_HISTORY.md` from the git log (I can create that now),
- add sanitizer targets, or
- commit a small deterministic `.mat` file into `tests/data/` and add a CTest that runs the demo on it.

---

Small note: this file is intended as a living, developer-facing guide. When you change build wiring (CMake modularization, new targets, or DataLoader API) please update this document so it reflects the current project conventions.

## Copilot & linter guidance (how to generate edits safely)

This section tells autocomplete/code-generation tools (and contributors using them) how to produce edits that play nicely with the project's linters, CMake, and build flags. Follow these concrete rules when generating or editing code in this repository.

- Language & standard
    - Target C++20. The project sets `CMAKE_CXX_STANDARD 20` in `Flags.cmake`.

- Compiler flags and checks
    - Respect the compiler flags in `Flags.cmake` (e.g., `-Wall -Wpedantic -Wshadow`). Avoid constructs that intentionally silence those warnings.
    - Prefer explicit casts where needed to avoid narrowing or signed/unsigned comparison warnings.

- Formatting & static checks
    - Run `clang-format` on edits when adding or changing multiple lines of C++ code. Keep the existing style (four-space indentation, no mixed tabs).
    - Run `clang-tidy` fixes where appropriate (do not enable new `clang-tidy` checks that will fail CI unless the code is updated to satisfy them).

- Small patterns that cause linter warnings (avoid these)
    - Do not declare multiple variables in one statement. Instead prefer one declaration per line:

        // avoid
        std::vector<int> a, b;

        // prefer
        std::vector<int> a;
        std::vector<int> b;

    - Use uppercase unsigned suffixes for integer literals when needed (e.g., `42U` not `42u`) to match the style used in this project.
    - Always use braces for single-line control statements (loops/ifs) to avoid clang-tidy/clang-format complaints and accidental mistakes:

        // prefer
        for (auto &x : xs) {
            do_something(x);
        }

    - Avoid calling `emplace_back` in a loop without reserving capacity first. Either call `reserve()` or populate a pre-sized container.

- CMake editing rules
    - When editing CMake files: don't leave stray text fragments or unmatched parentheses — CMake errors can be introduced by accidental copy/paste.
    - Prefer extracting logical blocks into separate `.cmake` files and `include()` them from `Main.cmake` (the repository already uses this pattern).
    - When adding new top-level targets, ensure any source files needed by tests are linked into the test target so link-time errors do not occur.

- Tests and CI
    - Add or update tests using GoogleTest where relevant. Use `gtest_discover_tests()` to register test binaries with CTest.
    - After edits that touch CMake, run a local configure step (`cmake -S . -B build`) and run `ctest --test-dir build --output-on-failure -j 4` to validate behavior.

- Commit and PR hygiene
    - Keep generated changes minimal and targeted. When a copilot suggestion spans multiple files, re-run formatting and the test suite before committing.

Following these rules will keep Copilot (or any code-generation assistant) from producing code that triggers the project's compiler warnings or CMake parse errors. If you have an automated flow that edits many files, run the full build and tests locally after generation and fix any linter or build failures before pushing.

