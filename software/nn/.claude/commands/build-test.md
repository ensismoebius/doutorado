---
description: "Deterministic CMake/Ninja build and targeted test execution with release flags and ccache."
---

# build-test

Verify changes quickly with deterministic CMake/Ninja workflows.

## Project Context (nn framework)

**Presets** — always use, never invent raw cmake flags:
- `cmake --preset=max-performance` — CPU release build → `out/build/max-performance/`
- `cmake --preset=max-performance-opencl` — GPU/OpenCL build → `out/build/max-performance-opencl/`
- `cmake --preset=Clang_20.1.8_x86_64-pc-linux-gnu` — debug/sanitizer build

**Named targets** (use with `--target`):

| Target | What |
|---|---|
| `core_gtest` | All core unit tests |
| `experiment04` | Experiment 04 binary |
| `experiment04_lib` | Experiment 04 library only |
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

# GPU build + experiment04
cmake --preset=max-performance-opencl
cmake --build out/build/max-performance-opencl --target experiment04 -j$(nproc)

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
