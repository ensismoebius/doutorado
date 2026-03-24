---
name: agent-customization
description: "Skill to create, update, review and validate VS Code agent customization files (instructions, prompts, agents, SKILL.md). Guides agents to prefer existing code, minimize redundancy, and follow repo conventions."
---

# agent-customization (repo-scoped skill)

Purpose
-------
Help an agent produce high-quality VS Code customization assets for this repository: `.instructions.md`, `.prompt.md`, `.agent.md`, `SKILL.md`, `copilot-instructions.md`, and related files. Emphasizes reuse of existing code, avoiding duplication, and aligning with repo conventions.

Assumptions
-----------
- This skill is intended for workspace-scoped use (team-level files under `.github/`), unless the user explicitly requests user-scoped output.
- The agent has read access to the repository and can run read-only scans (grep/Explore subagent) before editing.
- The project uses a formal style guide and CI; new public APIs require changelog entries and tests.

When to Use
-----------
- Create or update automation/instruction files for the repository.
- Troubleshoot why an instruction/skill is not loaded.
- Define `applyTo` patterns or create new custom agents.

Decision Flow (high level)
--------------------------
- Scope: workspace vs user? → workspace by default.
- Primitive: Instruction? Prompt? Skill? Agent? → choose by breadth:
  - Instruction: always-on guidance for the repo
  - Prompt: single-step, parameterized task
  - Skill: multi-step, bundled assets, reusable
  - Agent: multi-stage workflow that needs strict tool restrictions or context isolation
- Location: `.github/` for workspace artifacts; user folder for personal prompts.

Step-by-step Workflow
---------------------
1. Clarify scope with the user (workspace vs personal, quick fix vs reusable workflow).
2. Scan repository for existing customization files and related code (search `applyTo`, `copilot-instructions.md`, patterns, example prompts).
3. Identify reuse opportunities (existing utilities, headers, templates). Prefer reuse over adding new files.
4. Draft frontmatter YAML with `name`, `description`, `applyTo` (specific globs), and `usage` examples.
5. Write body following repository templates and this skill's guidance.
6. Add a short validation checklist and, if relevant, a small test or example invocation.
7. Run repository formatting (e.g., `clang-format -i` for changed C++ files) and lint checks if requested.
8. Commit with a descriptive message and add a `CHANGELOG.md` entry for public API or behavior changes.

Branching & Decision Points
---------------------------
- If change would add or change public APIs → require `CHANGELOG.md` and a test.
- If multiple similar helpers exist → refactor to reuse; propose a single helper location.
- If `applyTo` would be global (`"**"`) → prompt the user to confirm; prefer narrow globs.
- If YAML frontmatter or description contains ambiguous trigger phrases → ask the user for explicit keywords.

Quality Criteria / Completion Checks
-----------------------------------
- File placed in the correct folder (e.g., `.github/skills/<name>/` or `.github/` root).
- YAML frontmatter is syntactically valid (quoted values when needed).
- `description` contains trigger phrases the agent can match.
- `applyTo` uses specific globs unless explicitly requested global.
- New behavior changes have tests or a test plan.
- Code changes follow repository naming and style conventions.
- Formatting and basic CI checks run locally (or instructions provided to run them).

Agent Rules (how to apply this skill)
-------------------------------------
- Always scan the repository for existing implementations before adding new code.
- Prefer editing existing files or extracting a shared helper rather than duplicating logic.
- Ask clarifying questions if any of: scope, public API changes, or test requirements are ambiguous.
- Never add new public API surfaces without a `CHANGELOG.md` entry.
- When adding `applyTo`, favor narrow glob patterns; avoid `**` unless the user confirms.
- If the user requests web research: ask for permission, then fetch up to 6 authoritative references, summarize them, and highlight actionable practices to adopt.
- When the agent modifies C++ sources, run `clang-format -i` on changed files before committing.

Prompts / Example Invocations
----------------------------
- "Create a workspace SKILL.md that enforces our repo's code-reuse policy and adds examples."
- "Fix `.github/copilot-instructions.md` so it triggers on `applyTo: src/**` and add description keywords: "format|clang-format|naming"."
- "Draft a prompt to create a new helper in `src/core/` and include a unit test skeleton."

Suggested Output Structure
--------------------------
- Minimal header/frontmatter with `name` and `description`.
- Short rationale (1–2 lines) explaining why the change is needed.
- Concrete checklist for validation.
- Example commands to run locally (formatting, build/test) and a suggested commit message.

Example Commit Message
----------------------
- `chore: add agent customization skill for consistent prompts and instructions`

Next Steps / Questions for User
------------------------------
- Do you want this skill to be workspace-scoped (default) or user-scoped?
- Should I perform a web search now for state-of-the-art code-generation and prompt-engineering practices and append curated references?


---

Generated by agent-customization skill template. Update sections above as needed for your team's conventions.
