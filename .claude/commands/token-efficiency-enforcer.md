
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# token-efficiency-enforcer

Goal
- Minimize token usage while preserving actionable correctness.

Rules

- RULE: CONSTRAINT_FIRST
  DO: Use compact `RULE/DO/AVOID` blocks
  AVOID: No long narrative explanations by default
- RULE: OUTPUT_MINIMAL
  DO: Return structured and concise outputs unless detail is explicitly requested
  AVOID: No repeating unchanged context
- RULE: TOOL_HEAVY
  DO: Prefer local tools and short summaries over large raw dumps
  AVOID: No token-heavy copy/paste outputs
- RULE: BUDGET_RESPECTED
  DO: Use `get_context`/`get_source_range` (MCP) and read the `recommended_ranges` it gives back when a symbol exceeds budget
  AVOID: Never guess a smaller line range yourself when a request comes back `"truncated": true` — the recommended chunking is already computed

Validation

- Responses are concise, structured, and non-redundant.
- No repeated preamble or closing summaries that restate the task.
- Every output line is actionable or informational — no filler.

Project Context (nn framework)

**Project knowledge sources** (prefer over reading large source files):
- `CLAUDE.md` — contracts, invariants, build commands, key file index
- `.wiki/` — concepts, guides, experiment descriptions
- Use `navigation` skill first to locate files before reading them

**Efficient search patterns** — MCP first, `rg`/`jq` for what it doesn't cover:
- Find a symbol across all sources → `find_symbol`/`search_text` (MCP), not `rg 'LeakyBPTT' include/ src/ --type cpp -l`
- Run only matching tests → `run_tests(filter="core_gtest")` (MCP), or `ctest --test-dir out/build/max-performance -R core_gtest --output-on-failure`
- Check specific profile field without reading entire file → `jq '.model.paradigm' src/experiments/guayaquil/profiles/article-lstm-ae.json` (the MCP doesn't query arbitrary JSON — `jq` stays the right tool here)

**Avoid reading** entire files when `get_source_range`/`symbol_source` (MCP), a grep, or a `jq` query suffices.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Output Format Preferences

| Task | Preferred format |
|------|-----------------|
| Audit / findings | Bullet list with `[FILE] Issue / Impact / Fix` |
| Config changes | Minimal diff or inline key→value pairs |
| Search results | File:line references, not full file dumps |
| Explanations | 2–3 sentence summary + pointer to source |
