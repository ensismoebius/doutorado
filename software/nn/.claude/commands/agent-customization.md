---
description: "Create, update, review, and validate Claude Code command files (.claude/commands/*.md)."
---

# agent-customization

Create or update Claude Code command assets with minimal token usage and deterministic behavior.

## Rules

- **SCOPE_FIRST**: Default to project scope under `.claude/commands/`. Use `~/.claude/commands/` only when explicitly requested for user-wide scope.
- **REUSE_EXISTING**: Search existing commands before adding new files. Update in place when possible.
- **TIGHT_FRONTMATTER**: Keep frontmatter to `description` only unless required. Description must be specific enough to distinguish the command in a list.
- **EXPLICIT_CHECKS**: End every command with a short validation checklist. No long narrative explanations.
- **PERF_GATE**: For performance-sensitive code change commands, invoke `cpp-performance-and-consistency-optimizer` rules. No optimization guidance without measurement-first constraints.

## Artifact Type Detection

| User asks about | Action |
|----------------|--------|
| Slash command / skill | Create/update `.claude/commands/<name>.md` |
| Project-wide behavior | Consider `CLAUDE.md` addition instead |
| Recurring automation | Use `update-config` skill for hooks |

## Workflow

1. Detect artifact type (command / CLAUDE.md / hook).
2. Search `.claude/commands/` for existing files.
3. Edit in place when possible; create new file only when needed.
4. Keep guidance concise, imperative, and testable.

## Validation

- File in `.claude/commands/` with `.md` extension.
- Frontmatter has valid `description` field.
- Constraints are explicit (DO / No).
- No conflicting duplicate rules with existing commands.
