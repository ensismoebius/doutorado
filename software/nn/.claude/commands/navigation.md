---
description: "Locate files, symbols, tests, and targets quickly with minimal tool calls."
---

# navigation

Find the right file/symbol fast, with reproducible search steps.

## Rules

- **SEARCH_NARROWLY**: Start in `src/`, `include/`, `cmake/`, `scripts/`. Avoid `build/`, `_deps/`, generated outputs unless requested.
- **CHEAP_TO_DEEP**: Use file search → grep search → read file. Never read large files before locating symbol anchors.
- **USAGE_CONFIRMATION**: Use symbol usage tools for cross-file impact. Never assume from a single occurrence.
- **PERF_GATE**: When locating hot paths, include allocation and loop hotspots explicitly. Don't produce search plans that ignore memory/CPU cost centers.

## Workflow

1. Locate filenames with `find src/ include/ cmake/ scripts/ -name "*.hpp" -o -name "*.cpp"` or `rg --files`.
2. Locate symbols/text with `rg <symbol>`.
3. Read only the needed line ranges.
4. Confirm references across all usages and related tests.

## Validation

- Candidate list includes implementation + header + tests.
- Search scope excludes irrelevant directories (`build/`, `_deps/`).
