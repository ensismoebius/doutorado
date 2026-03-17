# copilot-instructions.md — Project guide for Copilot and automation

Purpose
-------
Authoritative guidance for automated code generation, refactors, and CI interactions for the `nn` repository (SNN + Autoencoder + EEG+Audio framework). Use this as the first reference when proposing structural changes (new classes, public APIs, CMake changes), formatting rules, or behavior-affecting edits.

When to update this file:
- Any public API change (headers, exported CMake targets)
- Changes to data format or protocol handling (.mat schema, dataset layout)
- New long-lived components (experiment classes, prefetcher redesigns)
- Tooling/formatting changes that affect the repo (e.g., `.clang-format`)

Key principles
- Prefer minimal, local changes when possible — avoid rippling API changes across the tree.
- Preserve existing public contracts unless a clear migration path and changelog entry are provided.
- Add tests for behavior-critical changes (data loaders, prefetcher, MAT I/O).
- Document performance-sensitive design decisions (OpenMP usage, thread model).

Overview of project conventions
------------------------------
- Language: C++20 (GNU toolchain on Linux is primary environment).
- Build system: CMake + Ninja; use `build/` for out-of-tree builds.
- Tests: GoogleTest; use `ctest` or `ninja` targets created by CMake.
- Formatting: `.clang-format` at repo root governs style — update this file when changing formatting preferences.

High-level modules and where to find them
- Core library: `src/core/` (layers, tensor, optimizers, dataLoaders).
- Experiments and demos: `src/experiments/` (experiment-specific CLIs, pipelines).
- Vendored code and CMake helpers: `cmake/` and `lib/`.

Important recent refactors
-------------------------
- `Experiment04`: The pipeline previously in `src/experiments/03/experiment04.cpp` has been moved into a class `Experiment04` with header in `src/experiments/03/lib/include/experiment04.hpp` and implementation in `src/experiments/03/lib/src/experiment04.cpp`. New code should prefer constructing and invoking the experiment via that class rather than copying pipeline logic into new mains.
- `BatchPrefetcher`: Reworked to single-producer threaded design using a bounded deque to serialize MAT I/O. This avoids concurrent `matio` reads which are not assumed thread-safe. Always pass a `DataLoader&` to the `BatchPrefetcher` constructor and ensure the `DataLoader` outlives the prefetcher.
- `dataset_info`: Added `printDatasetSummary` in `src/experiments/03/lib/src/dataset_info.cpp`. By default it prints a fast `AudioWithEEG` estimate (`min(audio_rows, eeg_rows)`). Add a `--exact-summary` flag and associated logic if exact counts are required (be mindful of the extra I/O cost).
- `progress`: A single-line in-place progress helper was added in `src/experiments/03/lib/src/progress.cpp` — it centralizes effective-total calculation and display capping.
- Formatter change: `.clang-format` was updated to allow more aggressive breaking of return types and parameter lists; new code should follow the repo style by running `clang-format -i` on modified files.

How Copilot should act in this repo
----------------------------------
When generating or modifying code, always:

1. Check for existing public headers and common utilities (look in `include/` and `src/core/`).
2. Prefer adding code to experiment libraries under `src/experiments/<N>/lib` and exposing only necessary headers in that `lib/include` directory.
3. Update CMake targets via the `src/experiments/<N>/CMakeLists.txt` when adding new sources; prefer appending to existing library targets rather than creating many tiny libraries.
4. Add or update tests under `src/core/*/tests` or `src/experiments/*/tests` for non-trivial behavior changes.
5. Run `clang-format -i` on changed files before committing; CI checks formatting.

Threading and I/O constraints
-----------------------------
- MAT I/O: `matio` usage is centralized through `src/core/dataLoaders/mat_file.cpp` and higher-level dataset classes. The codebase assumes MAT reads must be serialized or mediated by a single producer thread (see `BatchPrefetcher`). Do not parallelize direct MAT reads across threads without an explicit session and careful synchronization.
- OpenMP: Use `OpenMP::OpenMP_CXX` targets from CMake; avoid introducing non-deterministic multithreading in experiments where reproducibility matters. Document OMP thread settings in experiment configs when necessary.

Public APIs and invariants to preserve
-------------------------------------
- `Protocol101117Dataset` must retain its `get_sample`, `get_item`, `collate`, and `set_input_mode` semantics. Other code depends on these semantics heavily.
- `DataLoader` iterator semantics: iterators capture a snapshot of indices for the current epoch; do not modify sampler state externally while iterators are in use.
- `BatchPrefetcher` expects a `DataLoader&` reference and will spawn a background producer thread. The `DataLoader` must outlive the `BatchPrefetcher` instance.

Formatting and linting
---------------------
- The repository uses a `.clang-format` config at the root. Recent updates allow more aggressive breaking of return types and parameter lists; always run `clang-format -i` on changed files.
- `clang-tidy` is configured via `.clang-tidy`; run it locally for larger patches to catch issues early.

Testing / CI notes
------------------
- Add GTest unit tests for critical behaviors (MAT I/O, DataLoader, BatchPrefetcher).
- CI uses sanitizer builds defined in `cmake/SanitizerFlags.cmake` — add tests or adjust flags if you introduce UB-prone code paths.

Commit, changelog and release policy
-----------------------------------
- Any public header change must be accompanied by a `CHANGELOG.md` entry and a rationale.
- For API-breaking changes: create a migration guide in `docs/` and bump semantic version accordingly.

Quick commands
--------------
- Configure & build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```
- Run tests:
```bash
ctest --test-dir build --output-on-failure -j4
```
- Format changed files:
```bash
git ls-files '*.cpp' '*.hpp' | xargs clang-format -i
```

Editing guidelines for the Experiment04 area
-----------------------------------------
- Do not duplicate the pipeline logic — instead extend `Experiment04` when adding instrumentation, metrics, or alternate flows.
- If you need new runtime options, add them to `src/experiments/03/lib/include/cli.hpp` and ensure `Experiment04` reads them from `Config`.
- Preserve `dataset_info` behavior; if you need exact counts, implement a lazy `computeExactAudioWithEEG()` behind a `--exact-summary` flag to avoid expensive default behavior.

If you want me to regenerate or expand this file with deeper API signatures, tests to add, or a proposed CHANGELOG entry, say what to focus on (API, docs, tests, CI).
