---
description: "Create, update, review, and validate Claude Code command files (.claude/commands/*.md)."
---

# agent-customization

Create or update Claude Code command assets with minimal token usage and deterministic behavior.

## Project Context (nn framework)

**Skill file source of truth:** `.claude/commands/` — edit here first

**Sync destinations** (via `scripts/dev/sync_cross_project_skills.sh`):
- `.github/skills/` — for GitHub Copilot
- `.opencode/skills/` — for OpenCode
- `~/.claude/commands/` — global Claude Code commands

**Convention:** Skill names match `/<skillname>` slash commands. File `build-test.md` → `/build-test`.

**`CLAUDE.md`** defines project-wide conventions that all skills must respect — check it before adding new constraints that may conflict.

**Adding a skill:** Create `.claude/commands/<skill-name>.md` with frontmatter `description:` field, then run `sync_cross_project_skills.sh` to propagate.

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
