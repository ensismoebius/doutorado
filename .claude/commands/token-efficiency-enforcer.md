
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

Validation

- Responses are concise, structured, and non-redundant.
- No repeated preamble or closing summaries that restate the task.
- Every output line is actionable or informational — no filler.

Project Context (nn framework)

**Project knowledge sources** (prefer over reading large source files):
- `CLAUDE.md` — contracts, invariants, build commands, key file index
- `.wiki/` — concepts, guides, experiment descriptions
- Use `navigation` skill first to locate files before reading them

**Efficient search patterns:**
```bash
# Find a symbol across all sources
rg 'LeakyBPTT' include/ src/ --type cpp -l

# Run only matching tests (avoid full suite)
ctest --test-dir out/build/max-performance -R core_gtest --output-on-failure

# Check specific profile field without reading entire file
jq '.model.paradigm' src/experiments/guayaquil/profiles/article-lstm-ae.json
```

**Avoid reading** entire files when a grep or `jq` query suffices.

**Wiki & knowledge graph:**
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
