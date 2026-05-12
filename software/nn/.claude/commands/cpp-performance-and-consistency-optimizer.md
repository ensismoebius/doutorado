---
description: "Profile-first C++20 optimization for speed, memory efficiency, consistency audits, and deterministic refactors."
---

# cpp-performance-and-consistency-optimizer

Improve latency/throughput and memory efficiency while preserving behavior. Detect consistency and architecture drift.

## Project Context (nn framework)

**Hot paths to profile first:**
1. `LifBPTT::forward` time loop — `(T*B, F)` shaped input; `v_mem_history`/`spike_history` pre-allocated outside loop
2. `Trainer` mini-batch loop — `BatchScope` wraps forward+backward; single `clFinish` per batch (OpenCL)
3. `LinearImpl::forward` / `matmul_rhs_transposed_bias` — fused OpenCL kernels available (1.70× speedup)

**Drift risks to check:**
- Never include `XTensorBackend` headers in `src/core/` targets — breaks backend agnosticism
- Time-major layout `(T*B, F)` must be preserved through all SNN layer transformations
- OpenCL fused kernels: `matmul_transposed_add_col_bias_relu/leaky_relu/sigmoid/tanh` in `OpenCLTensorBackend`

**Backend macro:** `#if defined(NN_BACKEND_OPENCL)` — guards GPU-only code paths

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

- **MEASURE_FIRST**: Profile before optimization. No premature micro-optimization.
- **HOTSPOT_FOCUS**: Optimize top hotspot functions first (top 5% by time/allocation). No broad low-impact tuning.
- **MEMORY_DOMINATES**: Prioritize cache locality, contiguous storage, low pointer chasing. No fragmented pointer graphs in hot paths.
- **MIN_ALLOCATIONS**: Reuse buffers, reserve capacity, avoid hot-path heap churn. No repeated allocations inside loops.
- **LOOP_ALLOCATION_BAN**: Allocate outside loops whenever possible. No heap allocation in inner loops.
- **DATA_LAYOUT**: Prefer SoA/contiguous layouts when vectorization matters. No AoS in SIMD-critical paths without reason.
- **MOVE_ZERO_COPY**: Use `const T&`/`T&&`, move on ownership transfer. No unnecessary deep copies.
- **RAII_OWNERSHIP**: Use stack objects or `std::unique_ptr`. No raw owning pointers.
- **LOOP_HYGIENE**: Hoist invariants, remove redundant recomputation. No expensive repeated work in inner loops.
- **ALGORITHM_FIRST**: Fix complexity class before micro-tuning. No micro-optimizing O(n²) bottlenecks.
- **COMPILER_FRIENDLY**: Use `constexpr`/`noexcept` where safe, inline-friendly small hot helpers. No unpredictable branching or virtual dispatch in hot loops.
- **OMP_HEAVY_ONLY**: Parallelize only heavy loops/batch operations, avoid false sharing. No OpenMP on tiny loops.

## Workflow

1. Profile: hotspots and allocations (`perf`, `valgrind/callgrind`).
2. Analyze: lifetimes, memory access patterns, contention.
3. Optimize: locality / layout / allocation / copy reductions.
4. Validate: same semantics, deterministic outputs, targeted tests.

## Mandatory Audit Format

```
[FILE]
Issue:
Type: (performance / architecture / bug)
Impact:
Fix:
```

## Non-Goals

- No semantic changes without explicit justification.
- No new external dependencies for optimization-only work.
- No over-engineered abstractions.
