---
name: validation-sequencing-enforcer
description: "Skill synchronized from Claude command /validation-sequencing-enforcer."
---

# validation-sequencing-enforcer

Converted from: .claude/commands/validation-sequencing-enforcer.md

# Validation Sequencing Enforcer

## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.

Enforce a deterministic validation order: fastest relevant checks first, then broader checks only when needed.

Use this skill whenever code, scripts, configs, or docs are changed.

## Objectives

1. Catch defects early with minimal latency
2. Reduce wasted runtime from unnecessary full-suite checks
3. Keep validation evidence clear and reproducible
4. Prevent unverified task closure

## Validation policy

Always validate in this order:

1. **Static and local check**
2. **Changed-scope behavioral check**
3. **Adjacent integration check**
4. **Broader system check** (only if risk justifies)

Do not jump to step 4 unless steps 1-3 pass or clearly cannot run.

## Required validation plan per task

Before running checks, define:

1. What changed
2. Which failure modes are most likely
3. Smallest command that can detect each failure mode
4. Escalation trigger for broader checks

## Enforced stages

### Stage 1: Fast static checks

- Syntax and parse checks for changed files
- Linting scoped to changed paths when available — `run_lint` (MCP, Python
  via `ruff`) or `get_violations`/`summarize_violations` (MCP, structural,
  cross-language: naming/docs/LOC/duplication, no build required) before
  reaching for a raw linter invocation
- Config validation commands for edited config files

If Stage 1 fails, fix first. Do not escalate yet.

### Stage 2: Targeted runtime checks

- Run only the script, function, or command path affected by the change —
  `run_tests(filter=<pattern>)` (MCP) is the deterministic, non-interactive
  form of this; falls back to raw `ctest -R`/`pytest -k` when the MCP isn't
  configured
- Prefer deterministic, non-interactive invocations
- Capture concise output that proves behavior — the MCP result's
  `status`/`counts`/`failed_tests` already is that; no need to re-parse a
  raw log for it

If Stage 2 fails, fix and re-run Stage 1 and Stage 2.

### Stage 3: Integration checks

- Validate direct neighbors and call paths impacted by the change — use
  `find_references`/`find_dependencies` (MCP) to enumerate them first
  rather than guessing which modules are "adjacent"; note the
  `"exact"`/`"heuristic"` tag on each row before trusting it
- Use focused service-level or module-level checks

Escalate to Stage 4 only if:

- Change has cross-cutting impact, or
- Risk is medium/high, or
- User explicitly requested broader validation

### Stage 4: Broader checks

- Repo-wide lint, test suites, or full health scripts — `run_lint`/
  `run_tests` (MCP, no filter) for a structured pass/fail summary instead
  of a raw log dump, or the project's own `analysis-all`/`ctest` directly
- Run only when justified by risk or explicit request

## Failure handling

When a check cannot be run, report:

1. Exact blocker
2. Impact on confidence
3. Best available substitute check

When a check fails, report:

1. Failing command
2. Key error lines
3. Fix applied
4. Re-run evidence

## Completion gate

Do not mark done unless all are true:

1. Fastest relevant checks were executed first
2. Targeted behavior was validated after edits
3. Escalation (or non-escalation) was justified
4. Any skipped checks were explicitly documented

## Anti-patterns to block

1. Running full-suite checks before scoped checks
2. Declaring success from static checks only when runtime changed
3. Reporting "looks good" without executed commands
4. Skipping re-validation after a fix

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual commands for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text; use it in Stage 3 to enumerate every call-site *shape* a change affects, not just named references
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Output standard

Completion report must include:

1. Validation sequence used (Stage 1-4)
2. Commands run at each executed stage
3. Result summary per stage
4. Residual risk and suggested next validation
