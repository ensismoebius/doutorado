---
name: agent-performance-enforcer
description: "Skill synchronized from Claude command /agent-performance-enforcer."
---

# agent-performance-enforcer

Converted from: .claude/commands/agent-performance-enforcer.md

# Agent Performance Enforcer

## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.

Enforce high-performance agent execution standards across planning, implementation, validation, and handoff.

Use this skill whenever work quality, speed, reliability, or context efficiency matters.

## Objectives

1. Improve first-pass success rate
2. Reduce unnecessary tool calls and token waste
3. Keep fixes deterministic and verifiable
4. Preserve repository safety and conventions

## Required input contract

Before implementation, ensure these are explicit:

1. Goal: one clear target outcome
2. Done criteria: concrete checks that define completion
3. Constraints: safety, compatibility, performance, style
4. Scope boundaries: what is in scope and out of scope

If any are missing, derive from repo docs and existing behavior. If still ambiguous, ask concise clarifying questions.

## Enforced workflow

### 1) Problem framing

- Restate the failure in one sentence
- Define expected behavior and measurable success
- Note assumptions and risks before edits

### 2) High-signal context only

- Read only relevant files and nearby references
- Prefer narrow, parallel searches over broad scans
- Avoid dumping large outputs unless needed for a decision

### 3) Task decomposition

- Split into small, testable deltas
- Apply one behavioral change at a time
- Validate after each delta where feasible

### 4) Deterministic implementation

- Prefer existing wrappers/scripts over ad-hoc commands — for anything
  about the code itself (symbol lookup, references, build/test/lint,
  git state), that means the `code_intelligence` MCP tools before a raw
  `grep`/`cmake`/`git` invocation; see the `navigation`/`build-test`/
  `patching` skills for which tool covers what
- Preserve established APIs, paths, and conventions
- Avoid hidden side effects

### 5) Fast feedback loop

- Run the smallest meaningful validation first
- Escalate to broader checks only as needed
- Stop and report clearly when validation cannot run

### 6) Repository hygiene

- Keep edits focused and minimal
- Do not reformat unrelated code
- Update docs for non-obvious behavior changes

### 7) Memory and recurrence control

- Record concise repo memory for root causes likely to recur
- Capture canonical fix patterns, not long narratives

### 8) Completion gate (must pass)

Do not declare done unless all apply:

- Behavior fixed against stated done criteria
- Relevant checks executed or blocked with reason
- Documentation updated when behavior changed
- Cross-agent skill sync completed when source skills changed
- Risks and next steps communicated clearly

## Anti-patterns to block

1. Silent fallback to alternate data source without stating limitation
2. Large speculative edits before reproducing the issue
3. Running expensive global checks before targeted validation
4. Relying on manual, non-repeatable steps when wrappers exist
5. Closing task without verification evidence

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual commands for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Output standard

When reporting completion, include:

1. What changed
2. Why it changed
3. How it was validated
4. Residual risks or follow-up actions
