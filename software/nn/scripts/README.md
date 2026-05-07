Clang-format pre-commit hook
=================================

This repository ships a small helper script and a template git-hook to run
`clang-format` on staged C/C++ files before commit.

Enable locally:

1. Make the helper executable:

```bash
chmod +x scripts/clang-format-changed.sh
```

2. Install the hook (local, not committed):

```bash
cp .githooks/pre-commit-template .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook will format staged C/C++ sources/headers and re-add them to the index
so the commit contains formatted files.

Notes:
- This approach keeps the hook untracked (git doesn't store `.git/hooks/*`).
- If you prefer a cross-platform/managed approach, use `pre-commit` and the
  `mirrors-clang-format` hook instead.

Cross-project skill sync
=================================

To make this repository's skill catalog available in other projects, run:

```bash
bash scripts/sync_cross_project_skills.sh
```

This publishes:
- `.github/skills/*` to `~/.copilot/skills/*`
- `.github/skills/*` to `~/.config/opencode/skills/*`
- `.claude/commands/*.md` to `~/.claude/commands/*.md`
- `.claude/commands/*.md` as skill files to `~/.claude/skills/*/SKILL.md`

Wiki export for Anytype
=================================

To prepare a clean Markdown import folder for Anytype, run:

```bash
bash scripts/export_wiki_for_anytype.sh --clean
```

Default output:
- `out/anytype/wiki/`

Useful options:
- `--source <dir>`: use a different wiki source path
- `--output <dir>`: write export to a custom folder
- `--include-graphify-out`: include generated graphify pages
- `--dry-run`: preview what would be copied

After running, import the output folder in Anytype:
- `Library -> Import -> Markdown & CSV`
