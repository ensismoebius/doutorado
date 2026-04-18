---
name: token-efficiency-enforcer
description: "Meta-skill to reduce verbosity while preserving precision and constraints."
---

# token-efficiency-enforcer

Goal
- Minimize token usage while preserving actionable correctness.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
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
