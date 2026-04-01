---
name: cpp-performance-and-consistency-optimizer
description: "Profile-first C++20 optimization skill for speed, memory efficiency, consistency audits, and deterministic refactors."
---

# cpp-performance-and-consistency-optimizer

Goal
- Improve latency/throughput and memory efficiency while preserving behavior.
- Detect consistency and architecture drift across the codebase.

Evidence
- Core Guidelines (performance/ownership): A
- Memory hierarchy + allocation/layout practices: B
- Overall confidence: A-

Rules
- RULE: MEASURE_FIRST
  DO: Profile before optimization.
  AVOID: Premature micro-optimization.
- RULE: HOTSPOT_FOCUS
  DO: Optimize top hotspot functions first (top 5% by time/allocation).
  AVOID: Broad low-impact tuning.
- RULE: MEMORY_DOMINATES
  DO: Prioritize cache locality, contiguous storage, low pointer chasing.
  AVOID: Fragmented pointer graphs in hot paths.
- RULE: MIN_ALLOCATIONS
  DO: Reuse buffers, reserve capacity, avoid hot-path heap churn.
  AVOID: Repeated allocations inside loops.
- RULE: LOOP_ALLOCATION_BAN
  DO: Allocate outside loops whenever possible.
  AVOID: Heap allocation in inner loops.
- RULE: DATA_LAYOUT
  DO: Prefer SoA/contiguous layouts when vectorization matters.
  AVOID: AoS in SIMD-critical paths without reason.
- RULE: MOVE_ZERO_COPY
  DO: Use `const T&`/`T&&`, move on ownership transfer.
  AVOID: Unnecessary deep copies.
- RULE: RAII_OWNERSHIP
  DO: Use stack objects or `std::unique_ptr` ownership.
  AVOID: Raw owning pointers.
- RULE: LOOP_HYGIENE
  DO: Hoist invariants, remove redundant recomputation.
  AVOID: Expensive repeated work in inner loops.
- RULE: ALGORITHM_FIRST
  DO: Fix complexity class before micro-tuning.
  AVOID: Micro-optimizing O(n^2) bottlenecks first.
- RULE: COMPILER_FRIENDLY
  DO: Use `constexpr`/`noexcept` (safe), inline-friendly small hot helpers.
  AVOID: unpredictable branching and virtual dispatch in hot loops.
- RULE: OMP_HEAVY_ONLY
  DO: Parallelize only heavy loops/batch operations and avoid false sharing.
  AVOID: OpenMP on tiny loops.

Workflow
1. Profile: hotspots and allocations (`perf`, `valgrind/callgrind`).
2. Analyze: lifetimes, memory access, and contention.
3. Optimize: locality/layout/allocation/copy reductions.
4. Validate: same semantics, deterministic outputs, targeted tests.

Mandatory Audit Format
```text
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
