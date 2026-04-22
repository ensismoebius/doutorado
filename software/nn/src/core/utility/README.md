# utility

Purpose
- Miscellaneous utilities used across the codebase (`progress`, batching helpers, synthetic data generators, etc.).

Usage
- These are low-level helpers; include the specific headers you need (see `include/nn/utility` for public headers).

Tests
- Unit tests under `src/core/utility/tests/` provide examples and expected behaviors.

Recent updates
- `create_batches` now writes directly into output batch tensors, removing intermediate per-batch tensor vectors and reducing copies.
- Batch container capacity is pre-reserved using the computed number of batches.
- Progress rendering now consumes logger lines incrementally via `drain_recent_lines()` to prevent repeated O(N) reprocessing during redraw.
- Added reusable signal preprocessing helpers in `include/nn/utility/SignalPreprocessing.hpp`:
	- `read_csv_signal(path) -> nn::Tensor` parses numeric CSV/TXT tokens into a column tensor.
	- `zscore_inplace(nn::Tensor&)` applies in-place z-score normalization with stable variance flooring.

Optimization techniques and references
- Direct tensor fill in `create_batches`: eliminate temporary vectors to reduce copy count and improve spatial locality on write paths (see [1], [2]).
- `batches.reserve(...)`: predictable container growth to avoid repeated reallocations and iterator/object moves (see [2]).
- Incremental log draining for progress UI: consume only new log events to reduce repeated per-refresh linear scans (see [3], [4]).

Bibliographic references
- [1] Kazushige Goto and Robert A. van de Geijn. Anatomy of High-Performance Matrix Multiplication. ACM TOMS, 34(3), 2008.
- [2] Scott Meyers. Effective Modern C++. O'Reilly, 2014.
- [3] Ulrich Drepper. What Every Programmer Should Know About Memory. Red Hat, 2007.
- [4] Martin Kleppmann. Designing Data-Intensive Applications. O'Reilly, 2017.
