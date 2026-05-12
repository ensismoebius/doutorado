---
description: "Meta-skill: produce concise, constraint-first, structured outputs. Minimize verbosity without losing precision."
---

# token-efficiency-enforcer

Minimize token usage while preserving actionable correctness.

## Project Context (nn framework)

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
jq '.model.paradigm' src/experiments/04/profiles/article-lstm-ae.json
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
