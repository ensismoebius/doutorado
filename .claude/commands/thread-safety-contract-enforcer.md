
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# thread-safety-contract-enforcer

Goal
- Ensure concurrent data loading is safe by making thread-safety contracts explicit, preventing shared-state races, and validating producer/consumer boundaries.

Rules

- RULE: DATALOADER_NOT_SHARED
  DO: A `DataLoader` instance must not be accessed from multiple threads concurrently. If parallel data loading is needed, each thread must own its own `DataLoader`
  AVOID: No implicit shared `DataLoader`
- RULE: PREFETCHER_OWNERSHIP
  DO: `BatchPrefetcher` owns exactly one producer thread. The consumer (training loop) must not outlive the prefetcher, and must call `stop()` before destruction
  AVOID: No detached threads
- RULE: SPSC_SINGLE_PRODUCER
  DO: `HighPerfSpscQueue` is single-producer-single-consumer. Assert that only one thread calls `push()` and only one calls `pop()`
  AVOID: No multi-producer use of SPSC queue
- RULE: EXCEPTION_SAFETY
  DO: `BatchPrefetcher` producer thread must catch all exceptions and signal shutdown rather than letting them propagate across thread boundaries (undefined behavior)
  AVOID: No uncaught exceptions in worker threads
- RULE: DESTRUCTOR_ORDER
  DO: Enforce destruction order: stop prefetcher thread before destroying the dataset/DataLoader it references. Document this in class comments
  AVOID: No use-after-free from worker thread outliving data source
- RULE: SANITIZER_CLEAN
  DO: New concurrent code must be tested under ThreadSanitizer (TSan) before merging
  AVOID: No concurrent primitives added without TSan validation

Validation

- `DataLoader` header documents thread-safety guarantee (thread-compatible or thread-safe).
- `BatchPrefetcher` destructor calls `stop()` and `join()` before returning.
- SPSC queue has a static assertion or runtime check for single-producer use.
- TSan reports zero data races on the prefetch path.

Project Context (nn framework)

**`OpenCLContext::s_batch_depth`** — plain `int` (NOT `thread_local`, NOT atomic). Single-threaded GPU dispatch assumed: only one thread drives the OpenCL command queue. Never access this from multiple threads.

**`ProgressManager`** — uses `std::mutex` for thread-safe progress updates. The Trainer calls it from the training thread; a display thread may read it concurrently. Contract: always lock before read or write.

**`DataLoader`** SPSC queue ownership model:
- One producer thread (`BatchPrefetcher`) writes batches into the queue
- One consumer thread (training loop) reads batches
- No shared `DataLoader` across threads; each thread owns its own instance

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Key Files to Audit

- [include/nn/dataLoaders/runtime/BatchPrefetcher.hpp](include/nn/dataLoaders/runtime/BatchPrefetcher.hpp) — producer thread lifecycle and exception handling
- [include/nn/dataLoaders/runtime/DataLoader.hpp](include/nn/dataLoaders/runtime/DataLoader.hpp) — document thread-safety guarantee (or lack thereof)
- [include/nn/utility/HighPerfSpscQueue.hpp](include/nn/utility/HighPerfSpscQueue.hpp) — assert single-producer-single-consumer contract

TSan Build Command

```bash
cmake -S . -B build-tsan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build build-tsan --target <target> -j4
./build-tsan/<target>  # run under TSan
```
