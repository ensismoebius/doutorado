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

How Copilot (and automation) should behave in this repo
1. Search for existing public headers and utilities in `include/` and `src/core/` before adding new code.
2. Prefer putting reusable experiment code under `src/experiments/<N>/lib` and exposing only necessary headers from `lib/include`.
3. When adding sources, update the corresponding `src/experiments/<N>/CMakeLists.txt` to add them to the experiment library target.
4. Add tests for non-trivial behavior changes under `src/core/*/tests` or `src/experiments/*/tests`.
5. Run `clang-format -i` on all changed files before committing; CI will check formatting.

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
