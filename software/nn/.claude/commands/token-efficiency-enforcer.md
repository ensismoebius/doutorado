---
description: "Meta-skill: produce concise, constraint-first, structured outputs. Minimize verbosity without losing precision."
---

# token-efficiency-enforcer

Minimize token usage while preserving actionable correctness.

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

**Project knowledge sources** (prefer over reading large source files):
- `CLAUDE.md` — contracts, invariants, build commands, key file index
- `.wiki/` — concepts, guides, experiment descriptions
- Use `navigation` skill first to locate files before reading them

**Efficient search patterns:**
```bash
# Find a symbol across all sources
rg 'LifBPTT' include/ src/ --type cpp -l

# Run only matching tests (avoid full suite)
ctest --test-dir out/build/max-performance -R core_gtest --output-on-failure

# Check specific profile field without reading entire file
jq '.model.paradigm' src/experiments/guayaquil/profiles/article-lstm-ae.json
```

**Avoid reading** entire files when a grep or `jq` query suffices.

## Rules

- **CONSTRAINT_FIRST**: Use compact `RULE/DO/AVOID` blocks. No long narrative explanations by default.
- **OUTPUT_MINIMAL**: Return structured and concise outputs unless detail is explicitly requested. No repeating unchanged context.
- **TOOL_HEAVY**: Prefer local tools and short summaries over large raw dumps. No token-heavy copy/paste outputs.

## Output Format Preferences

| Task | Preferred format |
|------|-----------------|
| Audit / findings | Bullet list with `[FILE] Issue / Impact / Fix` |
| Config changes | Minimal diff or inline key→value pairs |
| Search results | File:line references, not full file dumps |
| Explanations | 2–3 sentence summary + pointer to source |

## Validation

- Responses are concise, structured, and non-redundant.
- No repeated preamble or closing summaries that restate the task.
- Every output line is actionable or informational — no filler.
