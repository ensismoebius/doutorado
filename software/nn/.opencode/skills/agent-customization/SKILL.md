---
name: agent-customization
description: "Skill to create, update, review and validate VS Code agent customization files (instructions, prompts, agents, SKILL.md)."
---

# agent-customization

Goal
- Create/update agent assets with minimal token usage and deterministic behavior.

Rules
- RULE: SCOPE_FIRST
  DO: Default to workspace scope under `.github/`.
  AVOID: User-scope output unless explicitly requested.
- RULE: REUSE_EXISTING
  DO: Search existing instructions/prompts/skills before adding files.
  AVOID: Duplicate policy text.
- RULE: TIGHT_FRONTMATTER
  DO: Keep frontmatter to `name` and `description` unless required.
  AVOID: Vague trigger wording.
- RULE: EXPLICIT_CHECKS
  DO: End with short validation checklist.
  AVOID: Long narrative explanations.
- RULE: PERF_GATE
  DO: For performance-sensitive code changes, invoke `cpp-performance-and-consistency-optimizer` rules.
  AVOID: Authoring optimization guidance without measurement-first constraints.

Workflow
1. Detect artifact type: instruction / prompt / skill / agent.
2. Search existing files in `.github/`.
3. Edit in place when possible.
4. Keep guidance concise, imperative, and testable.

Validation
- File in correct folder.
- Frontmatter valid.
- Constraints explicit.
- No conflicting duplicate rules.
