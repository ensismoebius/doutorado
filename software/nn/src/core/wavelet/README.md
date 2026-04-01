# wavelet

Purpose
- Wavelet transform utilities and helpers used by experiments for time-frequency analysis.

Usage
- Use the provided transforms and result containers when implementing wavelet-based preprocessing or analysis.

Tests
- See tests under `src/core/wavelet/tests/` for examples.

Recent updates
- Iterative decomposition now pre-reserves `tasks_for_next_level` capacity per mode (`2 * tasks.size()` for packet mode and `tasks.size()` for regular DWT mode) to reduce per-level allocation churn.

Optimization techniques and references
- Mode-aware task-vector reservation: capacity is derived from decomposition branching factor, reducing repeated growth/reallocation costs across levels (see [1], [2]).

Bibliographic references
- [1] Ulrich Drepper. What Every Programmer Should Know About Memory. Red Hat, 2007.
- [2] Paul R. Wilson, Mark S. Johnstone, Michael Neely, and David Boles. Dynamic Storage Allocation: A Survey and Critical Review. IWMM, 1995.
