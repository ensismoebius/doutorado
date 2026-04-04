PGO workflow (GCC)
===================

This document describes a minimal profile-guided optimization (PGO) workflow for the project using the provided CMake presets.

Steps
-----

1. Generate instrumented build (collect profile data)

   ```bash
   cmake --preset pgo-generate
   cmake --build --preset pgo-generate -j$(nproc)
   ```

   Run a representative workload that exercises the hot paths of your application. Example approaches:

   - Run end-to-end demos or dataset-driven experiment runners.
   - Use a custom script to replay typical inputs for several minutes.

   The compiler will produce profiling data files (GCC: `.gcda`) in object/build directories.

2. Produce final PGO-optimized build

   After collecting the profiling runs, reconfigure and build with the `pgo-use` preset:

   ```bash
   cmake --preset pgo-use
   cmake --build --preset pgo-use -j$(nproc)
   ```

Notes & tips
-----------

- The `pgo-generate` preset disables whole-project LTO to make profile collection straightforward. The final `pgo-use` preset enables LTO again for maximum runtime performance.
- Choose a representative workload — PGO amplifies the effect of realistic runs.
- You can change the compiler flags in `CMakePresets.json` if you prefer to direct profile output to a specific directory or to tweak optimization levels.

Troubleshooting
---------------

- If linking fails with the fast linker enabled, keep `NN_ENABLE_FAST_LINKER=OFF` in the presets (stable behaviour). Re-enabling the fast linker requires making vendored targets LTO-safe.

Advanced
--------

- For Clang, use `-fprofile-instr-generate` / `-fcoverage-mapping` and `llvm-profdata` / `llvm-cov` conversion steps — adapt presets accordingly.
