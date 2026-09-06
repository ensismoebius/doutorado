
# build-test

Goal
- Verify changes quickly with deterministic CMake/Ninja workflows.

Rules

- RULE: RELEASE_BENCH
  DO: Use release flags for performance runs (`-O3 -march=native -flto`)
  AVOID: Never use debug flags in benchmarks
- RULE: NINJA_CCACHE
  DO: Prefer Ninja generator and `ccache`
  AVOID: Avoid slow non-cached iterative builds
- RULE: TARGETED_BUILD
  DO: Build specific targets first
  AVOID: Avoid full rebuilds during iteration
- RULE: TEST_TARGETED
  DO: Run nearest tests first, then broaden if needed
  AVOID: Avoid expensive global test runs by default
- RULE: FAIL_VISIBLE
  DO: Use `--output-on-failure`
  AVOID: Never let test failures be silent
- RULE: PERF_GATE
  DO: When optimizing, run before/after microbench or representative smoke timings
  AVOID: Never claim speedup without comparative measurement
- RULE: REPRO_BUILD
  DO: Keep compiler/flags/configuration explicit in CMake presets or cache entries
  AVOID: Avoid host-dependent implicit flag drift

Validation

- Build succeeds for touched targets.
- At least one relevant test/smoke run is executed.

Project Context (nn framework)

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

**MCP `run_build`/`run_tests` — prefer once a preset is configured:**
`run_build(target=...)` and `run_tests(toolchain="cmake", filter=...)` find the
most-recently-configured `out/build/*/CMakeCache.txt` automatically (this
project's own preset layout is what `_find_cmake_build_dir` was written
for) and return `{status, error_count/counts, failed_tests, log_tail}`
instead of a raw log to re-read. `filter` is the `-R` regex; `target` is
the `--target` name from the table above.

Caveat: it always picks the *most recently built* preset. If two presets
are configured and you need the one that ISN'T the freshest (e.g. confirm
the CPU build still passes right after a GPU rebuild), run raw `cmake`/
`ctest` with an explicit `--test-dir out/build/<preset>` instead — the MCP
tool has no preset parameter to force a choice.

**Raw cmake/ctest — when you need a specific preset, or first configure:**
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

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --target <target> -j4
ctest --test-dir build --output-on-failure -j4
```
