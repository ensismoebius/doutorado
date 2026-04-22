---
name: build-test
description: "Skill for deterministic CMake/Ninja build and targeted test execution."
---

# build-test

Goal
- Verify changes quickly with deterministic CMake/Ninja workflows.

Rules
- RULE: RELEASE_BENCH
  DO: Use release flags for performance runs (`-O3 -march=native -flto`).
  AVOID: Debug flags in benchmarks.
- RULE: NINJA_CCACHE
  DO: Prefer Ninja generator and `ccache`.
  AVOID: Slow non-cached iterative builds.
- RULE: TARGETED_BUILD
  DO: Build specific targets first.
  AVOID: Full rebuilds during iteration.
- RULE: TEST_TARGETED
  DO: Run nearest tests first, then broaden if needed.
  AVOID: Expensive global test runs by default.
- RULE: FAIL_VISIBLE
  DO: Use failure-visible outputs (`--output-on-failure`).
  AVOID: Silent test failures.
- RULE: PERF_GATE
  DO: When optimizing, run before/after microbench or representative smoke timings.
  AVOID: Claiming speedup without comparative measurement.
- RULE: REPRO_BUILD
  DO: Keep compiler/flags/configuration explicit in CMake presets or cache entries.
  AVOID: Host-dependent implicit flag drift.

Commands
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --target <target> -j4
ctest --test-dir build --output-on-failure -j4
```

Validation
- Build succeeds for touched targets.
- At least one relevant test/smoke run is executed.
