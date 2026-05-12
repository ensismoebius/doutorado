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
