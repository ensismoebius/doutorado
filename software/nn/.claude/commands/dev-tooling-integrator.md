---
description: "Standardize use of local CLI tools (rg, jq, entr, ctags) to reduce manual complexity and token-heavy workflows."
---

# dev-tooling-integrator

Use installed tooling to minimize repetitive manual steps and avoid token-heavy workarounds.

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual git/cmake for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` / `replace_symbol` — structural findings, complexity hotspots, and hash-gated multi-site renames/edits
- `run_build` / `run_tests` / `run_lint` / `run_format` — structured build/test/lint output, not raw logs
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Project Context (nn framework)

**Installed tools and their roles:**
- `rg` (ripgrep) — primary code/text search; use instead of `find + grep`
- `clang-format` — via `scripts/dev/clang-format-changed.sh`; staged files only
- `ccache` — wired in CMake presets; clear with `cmake --build ... --target clean-cache`
- `ctest` — test runner; use `-R <pattern>` to target specific tests

**Static analysis:**
- `cmake --build ... --target analysis-all` — runs cppcheck + clang-tidy
- `python3 scripts/ci/validate_static_analysis.py --list-approved` — show suppression allowlist

## Rules

- **SEARCH_TOOLING**: Use `rg` for code/text search and file discovery. No slow broad `find` + `grep` combos when `rg` suffices.
- **JSON_TOOLING**: Use `jq` to validate/query JSON configs. No manual JSON parsing or string matching.
- **LOOP_AUTOMATION**: Use `entr` or equivalent for watch/rebuild loops. No re-running repetitive commands manually.
- **INDEX_SUPPORT**: Use `ctags`/symbol indexes for fast navigation where useful. No repeated whole-file reads when a symbol index is available.

## Common Invocations

```bash
# Fast symbol search
rg "class Tensor" include/ src/

# Query a JSON config field
jq '.batch_size' results/guayaquil/config.json

# Watch and rebuild on change
find src/ include/ -name "*.cpp" -o -name "*.hpp" | entr cmake --build build --target nn -j4

# Regenerate ctags index
ctags -R --c++-kinds=+p --fields=+iaS --extras=+q src/ include/
```

## Validation

- Proposed workflows use existing local tools first before reaching for heavier alternatives.
- No manual JSON parsing where `jq` applies.
