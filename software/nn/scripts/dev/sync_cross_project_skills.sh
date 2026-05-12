#!/usr/bin/env bash
# sync_cross_project_skills.sh — Sync agent skill files to global config dirs.
#
# Copies skill definitions from the repo's canonical locations to the
# per-user config directories used by each agent:
#
#   Source (repo)                      Destination (user config)
#   .github/skills/*/SKILL.md      →  ~/.copilot/skills/
#   .claude/commands/*.md           →  ~/.claude/commands/
#                                   →  ~/.claude/skills/
#   (same .github skills)           →  ~/.config/opencode/skills/
#
# Usage:
#   scripts/dev/sync_cross_project_skills.sh [--dry-run]
#
# Run after editing any skill file in .claude/commands/ or .github/skills/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GITHUB_SKILLS_SRC="${REPO_ROOT}/.github/skills"
CLAUDE_COMMANDS_SRC="${REPO_ROOT}/.claude/commands"

COPILOT_SKILLS_DST="${HOME}/.copilot/skills"
OPENCODE_SKILLS_DST="${HOME}/.config/opencode/skills"
CLAUDE_COMMANDS_DST="${HOME}/.claude/commands"
CLAUDE_SKILLS_DST="${HOME}/.claude/skills"

sync_skills_dir() {
  local src_root="$1"
  local dst_root="$2"

  mkdir -p "${dst_root}"
  for skill_dir in "${src_root}"/*; do
    if [[ ! -d "${skill_dir}" ]]; then
      continue
    fi

    local skill_name
    skill_name="$(basename "${skill_dir}")"
    if [[ "${skill_name}" == "user" ]]; then
      continue
    fi

    local src_skill_file="${skill_dir}/SKILL.md"
    if [[ ! -f "${src_skill_file}" ]]; then
      continue
    fi

    local dst_skill_dir="${dst_root}/${skill_name}"
    mkdir -p "${dst_skill_dir}"
    cp -f "${src_skill_file}" "${dst_skill_dir}/SKILL.md"
  done
}

sync_claude_commands() {
  mkdir -p "${CLAUDE_COMMANDS_DST}"
  for cmd_file in "${CLAUDE_COMMANDS_SRC}"/*.md; do
    if [[ -f "${cmd_file}" ]]; then
      cp -f "${cmd_file}" "${CLAUDE_COMMANDS_DST}/$(basename "${cmd_file}")"
    fi
  done
}

build_claude_skill_from_command() {
  local command_file="$1"
  local skill_name="$2"
  local out_file="$3"

  python3 - "$command_file" "$skill_name" "$out_file" <<'PY'
from pathlib import Path
import re
import sys

command_file = Path(sys.argv[1])
skill_name = sys.argv[2]
out_file = Path(sys.argv[3])

text = command_file.read_text(encoding="utf-8")

if text.startswith("---\n"):
    parts = text.split("---\n", 2)
    if len(parts) >= 3:
        front, body = parts[1], parts[2]
        if re.search(r"(?m)^name:\s*", front) is None:
            front = f"name: {skill_name}\n" + front
        out = f"---\n{front}---\n{body}"
    else:
        out = f"---\nname: {skill_name}\ndescription: \"Mirrored from {command_file.name}\"\n---\n\n{text}"
else:
    out = f"---\nname: {skill_name}\ndescription: \"Mirrored from {command_file.name}\"\n---\n\n{text}"

out_file.write_text(out, encoding="utf-8")
PY
}

sync_claude_skills() {
  mkdir -p "${CLAUDE_SKILLS_DST}"

  for cmd_file in "${CLAUDE_COMMANDS_SRC}"/*.md; do
    if [[ ! -f "${cmd_file}" ]]; then
      continue
    fi

    local skill_name
    skill_name="$(basename "${cmd_file}" .md)"
    local dst_skill_dir="${CLAUDE_SKILLS_DST}/${skill_name}"
    mkdir -p "${dst_skill_dir}"
    build_claude_skill_from_command "${cmd_file}" "${skill_name}" "${dst_skill_dir}/SKILL.md"
  done
}

if [[ ! -d "${GITHUB_SKILLS_SRC}" ]]; then
  echo "ERROR: Missing source directory ${GITHUB_SKILLS_SRC}" >&2
  exit 1
fi

if [[ ! -d "${CLAUDE_COMMANDS_SRC}" ]]; then
  echo "ERROR: Missing source directory ${CLAUDE_COMMANDS_SRC}" >&2
  exit 1
fi

sync_skills_dir "${GITHUB_SKILLS_SRC}" "${COPILOT_SKILLS_DST}"
sync_skills_dir "${GITHUB_SKILLS_SRC}" "${OPENCODE_SKILLS_DST}"
sync_claude_commands
sync_claude_skills

echo "Done: synchronized skills/commands for cross-project use."
echo "  Copilot: ${COPILOT_SKILLS_DST}"
echo "  OpenCode: ${OPENCODE_SKILLS_DST}"
echo "  Claude commands: ${CLAUDE_COMMANDS_DST}"
echo "  Claude skills: ${CLAUDE_SKILLS_DST}"