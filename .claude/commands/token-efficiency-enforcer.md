
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
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

**Wiki & knowledge graph** (concepts, papers, docs — not code symbols; use the MCP tools above for those):
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

Output Format Preferences

| Task | Preferred format |
|------|-----------------|
| Audit / findings | Bullet list with `[FILE] Issue / Impact / Fix` |
| Config changes | Minimal diff or inline key→value pairs |
| Search results | File:line references, not full file dumps |
| Explanations | 2–3 sentence summary + pointer to source |
