---
name: navigation
description: "Skill to locate files, symbols, tests, and targets quickly with minimal tool calls."
---

# navigation

Goal
- Find the right file/symbol fast, with reproducible search steps.

Rules
- RULE: SEARCH_NARROWLY
  DO: Start in `src/`, `include/`, `cmake/`, `scripts/`.
  AVOID: `build/`, `_deps/`, generated outputs unless requested.
- RULE: CHEAP_TO_DEEP
  DO: `file_search` -> `grep_search` -> `read_file`.
  AVOID: Reading large files before locating symbol anchors.
- RULE: USAGE_CONFIRMATION
  DO: Use symbol usage tools for cross-file impact.
  AVOID: Assumptions from a single occurrence.
- RULE: PERF_GATE
  DO: When locating hot paths, include allocation and loop hotspots explicitly.
  AVOID: Search plans that ignore memory/CPU cost centers.

Workflow
1. Locate filenames.
2. Locate symbols/text.
3. Read only needed ranges.
4. Confirm references and tests.

Validation
- Candidate list includes implementation + header + tests.
- Search scope excludes irrelevant directories.
