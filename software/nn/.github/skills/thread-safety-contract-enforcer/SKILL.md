---
name: thread-safety-contract-enforcer
description: "Enforce thread-safety contracts for DataLoader, BatchPrefetcher, and SPSC queue — document ownership and validate no data races."
---

# thread-safety-contract-enforcer

Ensure concurrent data loading is safe by making thread-safety contracts explicit, preventing shared-state races, and validating producer/consumer boundaries.

## Rules

- **DATALOADER_NOT_SHARED**: A `DataLoader` instance must not be accessed from multiple threads concurrently. If parallel data loading is needed, each thread must own its own `DataLoader`. No implicit shared `DataLoader`.
- **PREFETCHER_OWNERSHIP**: `BatchPrefetcher` owns exactly one producer thread. The consumer (training loop) must not outlive the prefetcher, and must call `stop()` before destruction. No detached threads.
- **SPSC_SINGLE_PRODUCER**: `HighPerfSpscQueue` is single-producer-single-consumer. Assert that only one thread calls `push()` and only one calls `pop()`. No multi-producer use of SPSC queue.
- **EXCEPTION_SAFETY**: `BatchPrefetcher` producer thread must catch all exceptions and signal shutdown rather than letting them propagate across thread boundaries (undefined behavior). No uncaught exceptions in worker threads.
- **DESTRUCTOR_ORDER**: Enforce destruction order: stop prefetcher thread before destroying the dataset/DataLoader it references. Document this in class comments. No use-after-free from worker thread outliving data source.
- **SANITIZER_CLEAN**: New concurrent code must be tested under ThreadSanitizer (TSan) before merging. No concurrent primitives added without TSan validation.

## Key Files to Audit

- [include/dataLoaders/runtime/BatchPrefetcher.hpp](include/dataLoaders/runtime/BatchPrefetcher.hpp) — producer thread lifecycle and exception handling
- [include/dataLoaders/runtime/DataLoader.hpp](include/dataLoaders/runtime/DataLoader.hpp) — document thread-safety guarantee (or lack thereof)
- [include/utility/HighPerfSpscQueue.hpp](include/utility/HighPerfSpscQueue.hpp) — assert single-producer-single-consumer contract

## TSan Build Command

```bash
cmake -S . -B build-tsan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build build-tsan --target <target> -j4
./build-tsan/<target>  # run under TSan
```

## Validation

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
