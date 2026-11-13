---
name: static-analysis-enforcer
description: "Enforce cppcheck and flawfinder static analysis checks; remediate critical warnings before completion."
---

# static-analysis-enforcer

Keep code quality and safety regressions visible and controlled.

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

**Gates:** clang-tidy (`.clang-tidy`, `WarningsAsErrors: "*"`) and cppcheck (`analysis-cppcheck`, `--error-exitcode=1`) both fail on any finding; a false positive gets a one-line `NOLINT(<check>)` / `cppcheck-suppress <id>` with its reason. See `.wiki/Guides/Static-Analysis.md`.

**Config files:**
- `.clang-tidy` — clang-tidy checks at repo root


**CMake targets:**
- `analysis-cppcheck` — run cppcheck only
- `analysis-clang-tidy` — run clang-tidy only
- `analysis-all` — cppcheck + flawfinder + clang-tidy (fails if one is not installed)

```bash
run-clang-tidy -p out/build/max-performance 'software/nn/src/'   # must report nothing
cmake --build --preset=max-performance --target analysis-cppcheck # fails on any finding
```

## Rules

- **ANALYZE_ALWAYS**: Run `cppcheck` and relevant analyzer tasks on touched areas. No shipping unreviewed static-analysis findings.
- **SECURITY_SCAN**: Run `flawfinder` for security-sensitive paths. No ignoring high-risk warnings.
- **WARNING_POLICY**: Fix critical warnings and undefined-behavior risks before marking a task complete. No suppressing issues without justification.

## Commands

```bash
# cppcheck on modified files
cppcheck --enable=all --std=c++20 --suppress=missingInclude \
  --error-exitcode=1 <file-or-dir>

# flawfinder security scan
flawfinder --minlevel=3 src/ include/

# clang-tidy (if configured)
clang-tidy <file> -- -std=c++20 -I include/
```

## Workflow

1. Run `cppcheck` on all files touched by the current change.
2. Run `flawfinder` on security-sensitive paths (I/O, parsing, network).
3. Triage findings: fix critical/undefined-behavior issues, document suppressed low-impact ones.
4. Re-run analyzers to confirm zero critical findings.

## Validation

- Critical analyzer findings are resolved or explicitly justified with inline comments.
- No new `cppcheck` errors in modified files.
