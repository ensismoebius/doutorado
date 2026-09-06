---
name: dev-tooling-integrator
description: "Standardize use of local CLI tools to reduce manual complexity and token-heavy workflows."
---

# dev-tooling-integrator

Goal
- Use installed tooling to minimize repetitive manual steps.

Rules
- RULE: SEARCH_TOOLING
  DO: Use `rg` for code/text search and file discovery.
  AVOID: Slow broad scans.
- RULE: JSON_TOOLING
  DO: Use `jq` to validate/query JSON configs.
  AVOID: Manual JSON parsing.
- RULE: LOOP_AUTOMATION
  DO: Use `entr` or equivalent for watch/rebuild loops.
  AVOID: Re-running repetitive commands manually.
- RULE: INDEX_SUPPORT
  DO: Use `ctags`/symbol indexes for fast navigation where useful.
  AVOID: Repeated whole-file reads.

Validation
- Proposed workflows use existing local tools first.

Project Context (nn framework)
**Installed tools and their roles:**
- `rg` (ripgrep) — primary code/text search; use instead of `find + grep`
- `clang-format` — via `scripts/dev/clang-format-changed.sh`; staged files only
- `ccache` — wired in CMake presets; clear with `cmake --build ... --target clean-cache`
- `ctest` — test runner; use `-R <pattern>` to target specific tests

**Static analysis:**
- `cmake --build ... --target analysis-all` — runs cppcheck + clang-tidy
- `python3 scripts/ci/validate_static_analysis.py --list-approved` — show suppression allowlist
