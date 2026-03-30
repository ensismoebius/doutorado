## copilot-instructions.md — Project guide for Copilot and automation

Purpose
-------
Authoritative guidance for automated code generation, refactors, and CI interactions for the `software/nn` project (SNN + autoencoder + EEG+audio framework). Use this file as the first reference when proposing structural changes (new classes, public APIs, CMake changes), formatting rules, or behavior-affecting edits.

When to update this file
- Public API changes (headers, exported CMake targets)
- Changes to data formats or dataset schemas (.mat, cnpy, etc.)
- New long-lived components (experiment libraries, prefetcher redesigns)
- Tooling or formatting changes that affect contributor workflows (e.g., `.clang-format`)

Core principles
- Prefer small, local, behavior-preserving edits; avoid wide surface-area API changes without a migration plan.
- Preserve public contracts unless you provide a clear migration path, tests, and a `CHANGELOG.md` entry.
- Add tests for behavior-critical changes (data loaders, prefetchers, MAT I/O).
- Document performance-sensitive decisions (OpenMP, threading model, I/O constraints).

Project conventions (quick)
- Language: C++20 (primary development on Linux). Keep APIs stable and documented.
- Build: CMake + Ninja; out-of-tree builds under `build/`.
- Tests: GoogleTest; run via `ctest` or ninja test targets.
- Formatting: `.clang-format` in repo root; run `clang-format -i` before committing.
- Linting: `clang-tidy` checks are available; run locally on large patches.

Where key code lives
- Core library: `src/core/` (layers, tensor, optimizers, dataLoaders).
- Experiments & CLI: `src/experiments/` (each experiment has a `lib/include` and `lib/src` for reusable components).
- Demos: `src/demos/`.
- CMake helpers and vendored patches: `cmake/`, `build/` subprojects.

Highlights from `docs/` (authoritative):
- The project documents methodology and reproducibility in `docs/docs.md` and naming rules in `docs/naming-conventions.md`. Follow those when adding experiments or refactoring.

Recent notable refactors (keep in mind)
- `Experiment04` — pipeline moved into `src/experiments/03/lib/include/experiment04.hpp` and `src/experiments/03/lib/src/experiment04.cpp`. Prefer extending this class rather than duplicating pipeline code in new mains.
- `BatchPrefetcher` — reworked to a single-producer, bounded-queue design to serialize `matio` reads. Always construct with a `DataLoader&` and ensure the `DataLoader` outlives the prefetcher.
- `dataset_info` — `printDatasetSummary` provides a fast `AudioWithEEG` estimate; add a guarded `--exact-summary` flag if you need exact counts (expensive I/O).
- `progress` — in-place, single-line progress helper centralizes effective-total logic and capping.
- Formatter change — `.clang-format` updated to allow more aggressive breaking of long signatures and return types. New code should follow this style.

Current project state
- Recent additions: `include/nn/testing/tempfile.hpp` (TempFile RAII helper), `include/nn/logging/Logger.hpp`, and a consolidated `include/nn/io/StateIO.hpp`.
- Tests & fixes: Added `src/core/serialization/tests/StateIO_gtest.cpp`; updated tests to be DB-independent and AddressSanitizer-clean; fixed `src/core/dataLoaders/SqliteBatchSource.cpp` prepared-stmt/windowing and migrated tests to use the tempfile helper.
- API ergonomics: Introduced PyTorch-like ergonomics (e.g., `model.to(device)`, `optimizer.step()` / `optimizer.zero_grad()`, `state_dict()` / `load_state_dict()`), plus Adam + model state roundtrip tests.
- Coverage & CI notes: An instrumented coverage build (`build-coverage`) is used for HTML reports. We observed `lcov/geninfo` "inconsistent: mismatched end line" errors caused by mismatches between gtest `TEST()` macro declaration lines and the compiled `TestBody()` DWARF ranges; the current pragmatic workflow is to perform a clean instrumented rebuild, run tests to regenerate `.gcda`, and use `lcov --ignore-errors inconsistent` plus selective `lcov --remove` filtering (e.g., `/usr/*`, `*/_deps/*`) when strict capture fails. Per-TU diagnostics are available for stricter investigations.

How Copilot (and automation) should behave in this repo
1. Search for existing public headers and utilities in `include/` and `src/core/` before adding new code.
2. Prefer putting reusable experiment code under `src/experiments/<N>/lib` and exposing only necessary headers from `lib/include`.
3. When adding sources, update the corresponding `src/experiments/<N>/CMakeLists.txt` to add them to the experiment library target.
4. Add tests for non-trivial behavior changes under `src/core/*/tests` or `src/experiments/*/tests`.
5. Run `clang-format -i` on all changed files before committing; CI will check formatting.
6. When modifying code in any `src/core/<library>/` implementation directory, update the corresponding `src/core/<library>/README.md` to summarize the change, document API or CMake impacts, and add or update unit tests or examples as appropriate. If the change affects a public API, include a `CHANGELOG.md` entry describing the change and migration guidance.

Threading, I/O, and safety constraints
- MAT I/O: All direct `matio` reads must be serialized. Use `BatchPrefetcher` or other mediator to avoid concurrent reads. Do not introduce parallel `matio` access unless you provide an explicit session-management layer and tests.
- OpenMP: Use `OpenMP::OpenMP_CXX` CMake targets; document OMP thread settings in experiment configs for reproducibility.

Public APIs and invariants to preserve
- `Protocol101117Dataset`: preserve `get_sample`/`get_item`/`collate` and `set_input_mode` semantics unless you provide a migration and tests.
- `DataLoader` iterator semantics: iterators capture epoch indices; do not mutate sampler state while iterators are active.
- `BatchPrefetcher` contract: constructed with `DataLoader&`, spawns a background producer, and requires the loader to outlive the prefetcher.

Formatting, linting and CI
- Always run `clang-format -i` on changed files locally. CI enforces formatting.
- Run `clang-tidy` for larger refactors to catch likely issues early.
- CI includes sanitizer builds (see `cmake/SanitizerFlags.cmake`) — verify code safety when changing low-level behavior.

Testing guidance
- Add GTest unit tests for critical behaviors: MAT I/O, `DataLoader`, `BatchPrefetcher`, dataset indexing and sampler behavior.
- Place tests near the code they validate (`src/core/.../tests` or `src/experiments/<N>/tests`).
 - Require a unit test for every new function added: place the test alongside the code it validates, cover normal and edge cases, and document non-obvious behavior or invariants in the test name or a short comment.

Commit, changelog and release
- Include a `CHANGELOG.md` entry for any public API change or meaningful behavior change.
- For API-breaking changes: provide a migration guide in `docs/` and include tests demonstrating the migration where feasible.

Quick commands
```bash
# Configure & build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure -j4

# Format changed files
git ls-files '*.cpp' '*.hpp' '*.h' | xargs -r clang-format -i
```

Naming conventions (summary from `docs/naming-conventions.md`)
- Types (`class`, `struct`, `enum`): `PascalCase`.
- Functions and methods: `snake_case`.
- Variables (local/member/parameter): `snake_case`.
- Constants: `kCamelCase` for `constexpr` values.
- Macros/include guards: `UPPER_SNAKE_CASE`.

Experiment04-specific editing guidance
- Extend `Experiment04` rather than copying its pipeline into new mains; if you need new runtime options add them to `src/experiments/03/lib/include/cli.hpp` and read them through the `Config` object.
- Keep `dataset_info`'s default fast estimate behavior; implement an explicit `--exact-summary` switch if exact row counts are required.

If you want a deeper regeneration of this file (expanded API signatures, suggested tests, or a proposed `CHANGELOG.md` entry), tell me which focus you prefer (API, docs, tests, CI) and I'll produce it.

Additional coding directives
---------------------------
- Avoid creating wrappers when unnecessary: prefer using the existing library code, classes, functions, properties, and other objects directly rather than introducing thin wrapper layers that increase indirection.
- Use temporary variables only when strictly necessary: favour concise expressions and direct usage; introduce temporaries when they improve readability, prevent repeated work, or clarify lifetime semantics.
- Prefer the pipeline programming style: compose transformations and small, well-named steps into readable pipelines rather than long imperative blocks when it improves clarity and maintainability.
- Name code objects intuitively and readably: choose descriptive, human-friendly names for functions, variables, classes, and structs so their purpose is immediately clear.
- Document everything: every function, variable, class, or struct should include a clear, concise comment describing its purpose, behaviour, inputs, outputs, and any important invariants or side effects.
- Do NOT add deprecated code tags when modifying the codebase: replace the symbol or object and refactor all references to point to the new symbol or object.

Preserve custom directives
-------------------------
- Do NOT remove or overwrite user-provided custom directives in this file. Automated edits must append or annotate but must not delete or rewrite existing custom sections unless explicitly requested by a human reviewer.
- When updating this file programmatically, retain exact text of existing `Additional coding directives` and any user-supplied paragraphs; prefer adding new guidance as new paragraphs or bullet items.
