---
name: update-agent-skills
description: "Synchronize all agent skill/command files across all three locations after code renames or API changes."
---

# update-agent-skills

Goal
- Keep all agent skill files consistent with the codebase after any rename, refactor, or new-layer addition.

Skill file locations (three must always stay in sync):
- `~/.claude/commands/*.md` — global Claude commands (all projects)
- `.claude/commands/*.md` — project-level Claude commands
- `.opencode/skills/<name>/SKILL.md` — OpenCode project skills

## When to run

- After any class rename (e.g. `LeakyImpl` → `LifImpl`)
- After any file rename (e.g. `Leaky.hpp` → `Lif.hpp`)
- After type alias changes in `include/layers/Layers.hpp`
- After adding new layer, loss, optimizer, or experiment
- After creating a new skill (add to all three locations)

## Workflow

1. Grep for stale names across all three skill directories.
2. Apply perl substitutions to all three locations (most-specific-first order to avoid double-replacement).
3. If creating a new skill: write `.claude/commands/<name>.md`, `.opencode/skills/<name>/SKILL.md`, copy to `~/.claude/commands/<name>.md`.
4. Verify zero stale references. Check file counts match.

## Rules

- RULE: MOST_SPECIFIC_FIRST
  DO: Replace `LeakyBPTTImpl` before `LeakyBPTT` before `Leaky`; prevents double-substitution.
  AVOID: Single broad replacement that hits unwanted substrings.

- RULE: WORD_BOUNDARY_SAFETY
  DO: Use perl `\b` word boundaries and negative lookahead `(?!ReLU)` when renaming `Leaky`.
  AVOID: `sed s/Leaky/Lif/g` — hits `LeakyReLU`.

- RULE: PROSE_PRESERVE
  DO: After mass rename, restore "Leaky Integrate-and-Fire" (neuroscience term) wherever it became "Lif Integrate-and-Fire".
  AVOID: Renaming the biological concept name — only the C++ class changes.

- RULE: THREE_LOCATIONS
  DO: Always update global (`~/.claude/commands/`), project (`.claude/commands/`), and OpenCode (`.opencode/skills/`) in the same operation.
  AVOID: Updating only one location and leaving the others stale.

- RULE: PYTHON_EXEMPT
  DO: Skip `.py` files — `snn.Leaky()` is SNNTorch's API, not this project's class.
  AVOID: Running perl rename on Python source files.

## Key renames applied (nn project history)

| Old | New | Date |
|---|---|---|
| `LeakyImpl` | `LifImpl` | 2026-05-12 |
| `LeakyBPTTImpl` | `LifBPTTImpl` | 2026-05-12 |
| `LeakyIntegratorImpl` | `LifIntegratorImpl` | 2026-05-12 |
| `nn::Leaky` | `nn::Lif` | 2026-05-12 |
| `nn::LeakyBPTT` | `nn::LifBPTT` | 2026-05-12 |
| `nn::LeakyIntegrator` | `nn::LifIntegrator` | 2026-05-12 |
| `Leaky.hpp` | `Lif.hpp` | 2026-05-12 |
| `LeakyBPTT.hpp` | `LifBPTT.hpp` | 2026-05-12 |
| `LeakyIntegrator.hpp` | `LifIntegrator.hpp` | 2026-05-12 |

## Current nn layer name reference

| Class | File | Alias |
|---|---|---|
| `LifImpl<Backend>` | `include/layers/spiking/Lif.hpp` | `nn::Lif` |
| `LifBPTTImpl<Backend>` | `include/layers/spiking/LifBPTT.hpp` | `nn::LifBPTT` |
| `LifIntegratorImpl<Backend>` | `include/layers/spiking/LifIntegrator.hpp` | `nn::LifIntegrator` |
| `LeakyReLUImpl<Backend>` | `include/layers/activations/LeakyReLU.hpp` | `nn::LeakyReLU` ← **do not rename** |

## Validation commands

```bash
# Zero stale old names
grep -r "LeakyImpl\|LeakyBPTTImpl\|\bLeaky\b" \
  ~/.claude/commands/ .claude/commands/ .opencode/skills/ \
  2>/dev/null | grep -v node_modules | grep -v LeakyReLU

# File counts match
echo "global:  $(ls ~/.claude/commands/*.md | wc -l)"
echo "project: $(ls .claude/commands/*.md | wc -l)"
echo "opencode:$(find .opencode/skills -name 'SKILL.md' | grep -v node_modules | wc -l)"
```
