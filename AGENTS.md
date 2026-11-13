# AGENTS.md — OpenCode guidance for nn project

## Build & Test Commands

```bash
# Using CMake presets (recommended)
cmake --preset=max-performance
cmake --build --preset=max-performance -j$(nproc)
ctest --test-dir out/build/max-performance --output-on-failure -j4

# Alternative: manual build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -j4

# Format check
clang-format --check src/**/*.cpp src/**/*.hpp include/**/*.hpp
```

## Critical Conventions

- **Language**: C++20, Linux only
- **Build**: CMake + preset `max-performance` (O3 + LTO + march=native)
- **Tests**: GoogleTest, run via `ctest` or ninja test targets
- **Formatting**: `.clang-format` in repo root; run `clang-format -i` before committing

## Directory Ownership

| Path | Purpose |
|------|---------|
| `src/core/` | Core library (layers, tensor, optimizers, dataLoaders) |
| `src/experiments/<name>/` | One binary per experiment (`thesis`, `meeting01`, `paraconsistentGA`, …), each with `lib/include`, `lib/src`, `profiles/`, `tests/` |
| `src/demos/` | Standalone demo applications |
| `include/` | Public headers, included without prefix (`"tensor/Tensor.hpp"`); there is no `include/nn/` |

## Important Gotchas

1. **Logging**: Use `nn::logging::Logger` (`NN_LOG_INFO`, `NN_LOG_ERROR`, etc.). Avoid `std::cout`/`std::cerr` in core code.
2. **MAT I/O**: All direct `matio` reads must be serialized. Use `BatchPrefetcher` to avoid concurrent reads.
3. **Header guards**: Some files use `#pragma once`, others use `#ifndef`. Don't change existing style.
4. **State I/O**: `include/io/StateIO.hpp` saves/loads state dicts; `include/serialization/` holds `NnSaver` and `NetworkSerializer` (`.npz`).

## Key APIs

- `nn::Tensor` — main tensor type
- `nn::layers::Linear`, `Leaky`, `ReLU` — core layers
- `nn::optimizers::Adam`, `SGD` — optimizers
- `model.state_dict()` / `model.load_state_dict()` — PyTorch-like serialization
- `optimizer.step()` / `optimizer.zero_grad()` — standard optimizer interface

## Testing

- Tests live in `src/core/*/tests/` or `src/experiments/<N>/tests/`
- GTest files typically named `*_gtest.cpp`
- Run specific test: `ctest --test-dir build -R <test-name>`

## OpenCode Skills Available

- `logging` — enforces `nn::logging::Logger` usage
- `build-test` — deterministic CMake/Ninja build patterns
- `nn-core-usage-enforcer` — reuse existing core abstractions
- `patching` — focused edit workflow with compile/test validation

## Existing Documentation

- **Wiki** (comprehensive): `.wiki/` directory - See `.wiki/Home.md` for table of contents
- Full guidance: `.github/copilot-instructions.md`