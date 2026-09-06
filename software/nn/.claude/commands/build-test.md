---
description: "Deterministic CMake/Ninja build and targeted test execution with release flags and ccache."
---

# build-test

Verify changes quickly with deterministic CMake/Ninja workflows.

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual git/cmake for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` / `replace_symbol` — structural findings, complexity hotspots, and hash-gated multi-site renames/edits
- `run_build` / `run_tests` / `run_lint` / `run_format` — structured build/test/lint output, not raw logs
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Project Context (nn framework)

**Presets** — always use, never invent raw cmake flags:
- `cmake --preset=max-performance` — CPU release build → `out/build/max-performance/`
- `cmake --preset=max-performance-opencl` — GPU/OpenCL build → `out/build/max-performance-opencl/`
- `cmake --preset=Clang_20.1.8_x86_64-pc-linux-gnu` — debug/sanitizer build

**Named targets** (use with `--target`):

| Target | What |
|---|---|
| `core_gtest` | All core unit tests |
| `guayaquil` | Experiment 04 binary |
| `guayaquil_lib` | Experiment 04 library only |
| `trainer_gtest` | Trainer/EpochResult/TrainerConfig tests |
| `profile_audit_gtest` | 25 profile-parsing validation tests |
| `nn_progress` | Progress bar library |
| `analysis-all` | All static analysis |
| `clean-cache` | Clear ccache |

**Quick-start patterns:**
```bash
# CPU build + core tests
cmake --preset=max-performance
cmake --build out/build/max-performance --target core_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R core --output-on-failure

# GPU build + guayaquil
cmake --preset=max-performance-opencl
cmake --build out/build/max-performance-opencl --target guayaquil -j$(nproc)

# Profile audit (after any profile JSON edit)
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

## Rules

- **RELEASE_BENCH**: Use release flags for performance runs (`-O3 -march=native -flto`). Never use debug flags in benchmarks.
- **NINJA_CCACHE**: Prefer Ninja generator and `ccache`. Avoid slow non-cached iterative builds.
- **TARGETED_BUILD**: Build specific targets first. Avoid full rebuilds during iteration.
- **TEST_TARGETED**: Run nearest tests first, then broaden if needed. Avoid expensive global test runs by default.
- **FAIL_VISIBLE**: Use `--output-on-failure`. Never let test failures be silent.
- **PERF_GATE**: When optimizing, run before/after microbench or representative smoke timings. Never claim speedup without comparative measurement.
- **REPRO_BUILD**: Keep compiler/flags/configuration explicit in CMake presets or cache entries. Avoid host-dependent implicit flag drift.

## Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --target <target> -j4
ctest --test-dir build --output-on-failure -j4
```

## Validation

- Build succeeds for touched targets.
- At least one relevant test/smoke run is executed.
