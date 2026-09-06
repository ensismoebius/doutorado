---
name: token-efficiency-enforcer
description: "Meta-skill to reduce verbosity while preserving precision and constraints."
---

# token-efficiency-enforcer

Goal
- Minimize token usage while preserving actionable correctness.

Rules
- RULE: CONSTRAINT_FIRST
  DO: Use compact `RULE/DO/AVOID` blocks.
  AVOID: Long narrative explanations by default.
- RULE: OUTPUT_MINIMAL
  DO: Return structured and concise outputs unless detail is requested.
  AVOID: Repeating unchanged context.
- RULE: TOOL_HEAVY
  DO: Prefer local tools and short summaries over large raw dumps.
  AVOID: Token-heavy copy/paste outputs.

Validation
- Responses are concise, structured, and non-redundant.

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
