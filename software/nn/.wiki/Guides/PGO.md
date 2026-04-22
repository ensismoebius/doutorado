# Profile-Guided Optimization (PGO) Workflow

This document describes a minimal PGO workflow for the nn project using CMake presets.

## Steps

### 1. Generate Instrumented Build

```bash
cmake --preset pgo-generate
cmake --build --preset pgo-generate -j$(nproc)
```

Run a representative workload that exercises hot paths:
- End-to-end demos or experiment runners
- Custom script to replay typical inputs

The compiler produces profiling data (`.gcda` files) in build directories.

### 2. Produce Final PGO-Optimized Build

```bash
cmake --preset pgo-use
cmake --build --preset pgo-use -j$(nproc)
```

## Notes

- `pgo-generate` disables whole-project LTO for straightforward profile collection
- `pgo-use` enables LTO again for maximum performance
- Choose representative workloads — PGO amplifies realistic run effects

## Troubleshooting

- If linking fails with fast linker: keep `NN_ENABLE_FAST_LINKER=OFF`
- Re-enabling fast linker requires LTO-safe vendor targets

## Advanced (Clang)

For Clang, use:
- `-fprofile-instr-generate` / `-fcoverage-mapping`
- `llvm-profdata` / `llvm-cov` conversion steps

## See Also

- [Architecture](../Architecture.md)
- [Build System](../Architecture.md) (for CMake presets)