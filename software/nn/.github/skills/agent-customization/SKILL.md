---
name: agent-customization
description: "Create, update, review, and validate Claude Code command files (.claude/commands/*.md)."
---

# agent-customization

Create or update Claude Code command assets with minimal token usage and deterministic behavior.

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

**Skill file source of truth:** `.claude/commands/` — edit here first

**Sync destinations** (via `scripts/dev/sync_cross_project_skills.sh`):
- `.github/skills/` — for GitHub Copilot
- `.opencode/skills/` — for OpenCode
- `~/.claude/commands/` — global Claude Code commands

**Convention:** Skill names match `/<skillname>` slash commands. File `build-test.md` → `/build-test`.

**`CLAUDE.md`** defines project-wide conventions that all skills must respect — check it before adding new constraints that may conflict.

**Adding a skill:** Create `.claude/commands/<skill-name>.md` with frontmatter `description:` field, then run `sync_cross_project_skills.sh` to propagate.

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
