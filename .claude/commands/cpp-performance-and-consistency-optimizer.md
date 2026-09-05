
# cpp-performance-and-consistency-optimizer

Goal
- Improve latency/throughput and memory efficiency while preserving behavior. Detect consistency and architecture drift.

Rules

- RULE: MEASURE_FIRST
  DO: Profile before optimization
  AVOID: No premature micro-optimization
- RULE: HOTSPOT_FOCUS
  DO: Optimize top hotspot functions first (top 5% by time/allocation)
  AVOID: No broad low-impact tuning
- RULE: MEMORY_DOMINATES
  DO: Prioritize cache locality, contiguous storage, low pointer chasing
  AVOID: No fragmented pointer graphs in hot paths
- RULE: MIN_ALLOCATIONS
  DO: Reuse buffers, reserve capacity, avoid hot-path heap churn
  AVOID: No repeated allocations inside loops
- RULE: LOOP_ALLOCATION_BAN
  DO: Allocate outside loops whenever possible
  AVOID: No heap allocation in inner loops
- RULE: DATA_LAYOUT
  DO: Prefer SoA/contiguous layouts when vectorization matters
  AVOID: No AoS in SIMD-critical paths without reason
- RULE: MOVE_ZERO_COPY
  DO: Use `const T&`/`T&&`, move on ownership transfer
  AVOID: No unnecessary deep copies
- RULE: RAII_OWNERSHIP
  DO: Use stack objects or `std::unique_ptr`
  AVOID: No raw owning pointers
- RULE: LOOP_HYGIENE
  DO: Hoist invariants, remove redundant recomputation
  AVOID: No expensive repeated work in inner loops
- RULE: ALGORITHM_FIRST
  DO: Fix complexity class before micro-tuning
  AVOID: No micro-optimizing O(n²) bottlenecks
- RULE: COMPILER_FRIENDLY
  DO: Use `constexpr`/`noexcept` where safe, inline-friendly small hot helpers
  AVOID: No unpredictable branching or virtual dispatch in hot loops
- RULE: OMP_HEAVY_ONLY
  DO: Parallelize only heavy loops/batch operations, avoid false sharing
  AVOID: No OpenMP on tiny loops

Workflow

1. Profile: hotspots and allocations (`perf`, `valgrind/callgrind`).
2. Analyze: lifetimes, memory access patterns, contention.
3. Optimize: locality / layout / allocation / copy reductions.
4. Validate: same semantics, deterministic outputs, targeted tests.

Project Context (nn framework)

**Hot paths to profile first:**
1. `LeakyBPTT::forward` time loop — `(T*B, F)` shaped input; `v_mem_history`/`spike_history` pre-allocated outside loop
2. `Trainer` mini-batch loop — `BatchScope` wraps forward+backward; single `clFinish` per batch (OpenCL)
3. `LinearImpl::forward` / `matmul_rhs_transposed_bias` — fused OpenCL kernels available (1.70× speedup)

**Drift risks to check:**
- Never include `XTensorBackend` headers in `src/core/` targets — breaks backend agnosticism
- Time-major layout `(T*B, F)` must be preserved through all SNN layer transformations
- OpenCL fused kernels: `matmul_transposed_add_col_bias_relu/leaky_relu/sigmoid/tanh` in `OpenCLTensorBackend`

**Backend macro:** `#if defined(NN_BACKEND_OPENCL)` — guards GPU-only code paths

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

**Wiki & knowledge graph** (concepts, papers, docs — not code symbols; use the MCP tools above for those):
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

Mandatory Audit Format

```
[FILE]
Issue:
Type: (performance / architecture / bug)
Impact:
Fix:
```

Non-Goals

- No semantic changes without explicit justification.
- No new external dependencies for optimization-only work.
- No over-engineered abstractions.
