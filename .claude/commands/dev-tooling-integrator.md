
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# dev-tooling-integrator

Goal
- Use installed tooling to minimize repetitive manual steps and avoid token-heavy workarounds.

Rules

- RULE: SEARCH_TOOLING
  DO: Use the code-intelligence MCP (`find_symbol`/`search_text`) for anything under `src/`/`include/`; `rg` only for what it doesn't index (configs, scripts, `build/`). For a structural question ("every 2-arg call to `foo`", regardless of formatting) use `ast_search`, not a hand-built regex.
  AVOID: No slow broad `find` + `grep` combos, no `rg` for a symbol lookup the MCP already answers structured and cheaper, and no regex asked to draw a distinction (argument count, nesting) that `ast_search`'s `$VAR`/`$$$VAR` patterns already draw for free
- RULE: JSON_TOOLING
  DO: Use `jq` to validate/query JSON configs
  AVOID: No manual JSON parsing or string matching
- RULE: LOOP_AUTOMATION
  DO: Use `entr` or equivalent for watch/rebuild loops
  AVOID: No re-running repetitive commands manually
- RULE: INDEX_SUPPORT
  DO: Use the code-intelligence MCP's persistent index (`list_symbols`/`find_symbol`/`outline_symbol`) for navigation
  AVOID: No `ctags` regeneration step and no repeated whole-file reads — the MCP index is already incremental and carries location + metrics + call graph, which `ctags` doesn't
- RULE: EXEC_TOOLING
  DO: Use `run_build`/`run_tests`/`detect_toolchain` (MCP) for a structured pass/fail result instead of re-parsing raw `cmake`/`ctest` output by hand
  AVOID: No copying a full build/test log into context when `{status, counts, failed_tests}` answers the question

Validation

- Proposed workflows use the code-intelligence MCP first for anything about code structure, `rg`/`jq`/`entr` for what it doesn't cover, before reaching for heavier alternatives.
- No manual JSON parsing where `jq` applies.
- No `ctags` regeneration proposed where the MCP index already covers the same navigation need.

Project Context (nn framework)

**Installed tools and their roles:**
- **code-intelligence MCP** — persistent, incremental symbol index; primary tool for "where is X", "what calls X", "what's in this file" — supersedes `ctags` and most `rg` symbol searches. `ast_search`/`ast_replace` add AST-pattern structural search and hash-gated rewrite (`$VAR`/`$$$VAR` patterns) across cpp/java/php/javascript/python — supersedes a hand-built regex for anything shaped like code structure, not text
- `rg` (ripgrep) — text search outside the index: config files, scripts, `build/`, anything not source the MCP tracks
- `clang-format` — via `scripts/dev/clang-format-changed.sh`; staged files only
- `ccache` — wired in CMake presets; clear with `cmake --build ... --target clean-cache`
- `ctest` — test runner; use `-R <pattern>` to target specific tests, or `run_tests(filter=...)` (MCP) for a structured result

**Static analysis:**
- `cmake --build ... --target analysis-all` — runs cppcheck + clang-tidy (C++-specific; the MCP's own `get_violations`/`summarize_violations` are a separate, cross-language, structural pass — naming/docs/LOC/duplication — run both, they catch different things)
- `python3 scripts/ci/validate_static_analysis.py --list-approved` — show suppression allowlist

Common Invocations

```bash
# Symbol search: prefer MCP find_symbol/search_text over this --
# rg "class Tensor" include/ src/ only for what the MCP doesn't index

# Query a JSON config field
jq '.batch_size' results/guayaquil/config.json

# Watch and rebuild on change
find src/ include/ -name "*.cpp" -o -name "*.hpp" | entr cmake --build build --target nn -j4
```
