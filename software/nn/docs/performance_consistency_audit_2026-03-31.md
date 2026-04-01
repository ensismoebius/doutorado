Codebase Performance and Consistency Audit
Date: 2026-03-31
Scope: `src/`, `include/` (repository code; excludes `build*/` artifacts)
Method: static scan for allocation patterns, logging inconsistencies, ownership drift, and hot-loop inefficiencies.

[FILE]
Issue: Mixed logging strategy in DB source initialization uses direct `std::cerr` in error paths while the module otherwise uses `NN_LOG_*`.
Type: architecture
Impact: Inconsistent output routing, harder progress-UI integration, duplicated logging behavior.
Fix: Replace direct `std::cerr` in `open_db()` failure branches with `NN_LOG_ERROR` and keep one logging channel.
Path: `src/core/dataLoaders/SqliteBatchSource.cpp`

[FILE]
Issue: Repeated dynamic growth of large accumulation buffers (`eeg_accum`, `audio_accum`) without capacity hints.
Type: performance
Impact: Reallocation churn and cache-unfriendly copies during trial decoding.
Fix: Pre-size/`reserve` accumulators using row-count*column-size metadata before row loops.
Path: `src/core/dataLoaders/SqliteBatchSource.cpp`

[FILE]
Issue: Per-window temporary `samp` vectors are built without `reserve` before repeated `insert` calls.
Type: performance
Impact: Frequent reallocations in hot window-construction loop.
Fix: Compute per-window expected size and call `samp.reserve(expected_cols)` once per iteration.
Path: `src/core/dataLoaders/SqliteBatchSource.cpp`

[FILE]
Issue: Batch construction performs extra temporary tensor vectors (`x_batch_vec`, `y_batch_vec`) before final copy into output batch tensors.
Type: performance
Impact: Avoidable copies and allocations per batch; increased RAM traffic.
Fix: Write directly into `x_batch`/`y_batch` inside index loop; remove intermediate vectors.
Path: `src/core/utility/batching.cpp`

[FILE]
Issue: Progress renderer polls `get_recent_lines(200)` each refresh and re-appends to local history every call.
Type: performance
Impact: O(N) repeated log handling per frame and potential duplicate processing under frequent updates.
Fix: Track last-consumed index or switch to drain-based incremental ingestion for redraw path.
Path: `src/core/utility/progress.cpp`

[FILE]
Issue: Level decomposition in wavelet code rebuilds task vectors each level without explicit capacity planning.
Type: performance
Impact: Repeated allocations in multi-level transforms.
Fix: Reserve `tasks_for_next_level` capacity based on mode (`2 * tasks.size()` for packet mode, `tasks.size()` for DWT mode).
Path: `src/core/wavelet/waveletOperations.cpp`

[FILE]
Issue: Configuration loader emits all parse/validation errors via `std::cerr` instead of centralized logger.
Type: architecture
Impact: Inconsistent observability and hard-to-control output behavior across experiments.
Fix: Migrate to `NN_LOG_ERROR` and preserve typed error prefixes.
Path: `src/experiments/Config.cpp`

[FILE]
Issue: WAV writer reports open-file failures to `std::cout` before throwing.
Type: architecture
Impact: Error channel misuse; inconsistent severity semantics and noisy stdout in pipelines.
Fix: Replace with `NN_LOG_ERROR` (or throw-only path if caller handles reporting).
Path: `src/core/wave/Wav.cpp`

[FILE]
Issue: `Linear` forward adds bias with nested scalar loops instead of vectorized backend operation.
Type: performance
Impact: Suboptimal CPU utilization and cache behavior on larger batch sizes.
Fix: Add backend-level rowwise bias broadcast/add and use it in `Linear::forward`.
Path: `include/nn/layers/Linear.hpp`

[FILE]
Issue: Ownership model drift in tests uses raw `new/delete` for tensor parameters.
Type: architecture
Impact: Diverges from RAII conventions and increases leak-risk in test failures.
Fix: Use stack objects or `std::unique_ptr<nn::Tensor>` plus non-owning views where needed.
Path: `src/core/optimizers/tests/optimizers_gtest.cpp`

[FILE]
Issue: Experiment pipelines are inconsistent in logging abstraction (some use logger, others direct console prints).
Type: architecture
Impact: Divergent runtime behavior, harder CI log parsing, inconsistent progress integration.
Fix: Standardize experiment entrypoints and pipelines on `NN_LOG_*` plus optional stream redirection.
Path: `src/experiments/00/phase00.cpp`, `src/experiments/02/Experiment02Pipeline.cpp`, `src/experiments/03/lib/src/experiment03.cpp`

[FILE]
Issue: Duplicate computation path in iterator dereference computes batch index vector before `fetch_batch()`, which recomputes the same indices.
Type: performance
Impact: Redundant work each dereference; unnecessary allocations and bounds checks.
Fix: Remove duplicate index construction from `operator*()` and rely on `fetch_batch()`.
Path: `src/core/dataLoaders/DataLoaderIterator.cpp`
