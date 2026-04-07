# AGENTS.md — OpenCode guidance for nn project

## Build & Test Commands

```bash
# Debug build (default)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure -j4

# Format check
clang-format --check src/**/*.cpp src/**/*.hpp include/**/*.hpp
```

## Critical Conventions

- **Language**: C++20, Linux only
- **Build**: CMake + Ninja (out-of-tree under `build/`)
- **Tests**: GoogleTest, run via `ctest` or ninja test targets
- **Formatting**: `.clang-format` in repo root; run `clang-format -i` before committing

## Directory Ownership

| Path | Purpose |
|------|---------|
| `src/core/` | Core library (layers, tensor, optimizers, dataLoaders) |
| `src/experiments/<N>/` | Each experiment has `lib/include` and `lib/src` |
| `src/demos/` | Standalone demo applications |
| `include/nn/` | Public headers |

## Important Gotchas

1. **Logging**: Use `nn::logging::Logger` (`NN_LOG_INFO`, `NN_LOG_ERROR`, etc.). Avoid `std::cout`/`std::cerr` in core code.
2. **MAT I/O**: All direct `matio` reads must be serialized. Use `BatchPrefetcher` to avoid concurrent reads.
3. **Header guards**: Some files use `#pragma once`, others use `#ifndef`. Don't change existing style.
4. **Duplicate files**: Two `StateIO.hpp` exist (`nn/serialization/` and `nn/io/`) — they're different implementations.

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

- Full guidance: `.github/copilot-instructions.md`
- Naming rules: `docs/naming-conventions.md`
- Project docs: `docs/docs.md`