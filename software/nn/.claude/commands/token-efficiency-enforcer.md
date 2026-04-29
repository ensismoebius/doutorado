---
description: "Meta-skill: produce concise, constraint-first, structured outputs. Minimize verbosity without losing precision."
---

# token-efficiency-enforcer

Minimize token usage while preserving actionable correctness.

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
