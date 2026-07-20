---
name: build-test
description: "Deterministic CMake/Ninja build and targeted test execution with release flags and ccache."
---

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

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --target <target> -j4
ctest --test-dir build --output-on-failure -j4
```
