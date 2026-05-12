---
description: "Synchronize all agent skill/command files across all three locations after code renames or API changes."
---

# update-agent-skills

Keep all agent skill files consistent with the current codebase after any rename, refactor, or API change.

## Skill file locations (three must always stay in sync)

| Location | Format | Scope |
|---|---|---|
| `~/.claude/commands/*.md` | Claude command (no `name:` field) | Global — all projects |
| `.claude/commands/*.md` | Claude command (no `name:` field) | Project-level |
| `.opencode/skills/<name>/SKILL.md` | OpenCode skill (`name:` + `description:` frontmatter) | Project-level |

All three sets contain the same skills. When one is updated, all three must be updated.

## When to run this skill

- After any **class rename** (e.g. `LeakyImpl` → `LifImpl`)
- After any **file rename** (e.g. `Leaky.hpp` → `Lif.hpp`)
- After any **type alias rename** in `include/layers/Layers.hpp`
- After adding a **new layer, loss, or optimizer** to the core library
- After adding a **new experiment** (update navigation and wiki skills)
- After **new skill creation** (add to all three locations)

## Procedure

### Step 1 — Identify stale references

```bash
# Find skills with old names (example: after a rename from FOO to BAR)
grep -rl "FOO" \
  ~/.claude/commands/ \
  .claude/commands/ \
  .opencode/skills/ \
  2>/dev/null | grep -v node_modules
```

### Step 2 — Apply substitutions to all three locations

Use perl for word-boundary safety (avoids partial matches like LeakyReLU):

```bash
for DIR in \
  "$HOME/.claude/commands" \
  ".claude/commands" \
  ".opencode/skills"; do
  find "$DIR" -name "*.md" ! -path "*/node_modules/*" | xargs perl -pi \
    -e 's/\bOLD_NAME\b/NEW_NAME/g;'
done
```

For multi-rename (most specific first, to avoid double-substitution):

```bash
for DIR in \
  "$HOME/.claude/commands" \
  ".claude/commands" \
  ".opencode/skills"; do
  find "$DIR" -name "*.md" ! -path "*/node_modules/*" | xargs perl -pi \
    -e 's/OldBPTTImpl/NewBPTTImpl/g;' \
    -e 's/OldImpl/NewImpl/g;' \
    -e 's/OldBPTT\.hpp/NewBPTT.hpp/g;' \
    -e 's/Old\.hpp/New.hpp/g;' \
    -e 's/OldBPTT/NewBPTT/g;' \
    -e 's/\bOld\b(?!SafeSuffix)/New/g;'
done
```

### Step 3 — Create a new skill (when adding a new one)

**Claude command** (`.claude/commands/<name>.md` and `~/.claude/commands/<name>.md`):

```markdown
---
description: "One-line description of what this skill does."
---

# <name>

<purpose paragraph>

## Rules

- Rule descriptions...

## Workflow

1. Steps...

## Validation

- Checks...
```

**OpenCode skill** (`.opencode/skills/<name>/SKILL.md`):

```markdown
---
name: <name>
description: "One-line description of what this skill does."
---

# <name>

Goal
- <purpose>

Rules
- RULE: <RULE_NAME>
  DO: ...
  AVOID: ...

Workflow
1. ...
```

Copy the new skill file to all three locations immediately after creation.

### Step 4 — Verify

```bash
# Zero stale old names
grep -r "OLD_NAME" \
  ~/.claude/commands/ .claude/commands/ .opencode/skills/ \
  2>/dev/null | grep -v node_modules

# Count files match across locations
echo "global:  $(ls ~/.claude/commands/*.md | wc -l)"
echo "project: $(ls .claude/commands/*.md | wc -l)"
echo "opencode:$(find .opencode/skills -name 'SKILL.md' | grep -v node_modules | wc -l)"
```

File counts should be equal (minus any global-only skills not relevant to this project).

## Key file index (nn project)

| What | Where |
|---|---|
| LIF neuron single-step | `include/layers/spiking/Lif.hpp` → `LifImpl<Backend>` |
| LIF neuron BPTT | `include/layers/spiking/LifBPTT.hpp` → `LifBPTTImpl<Backend>` |
| LIF integrator readout | `include/layers/spiking/LifIntegrator.hpp` → `LifIntegratorImpl<Backend>` |
| Layers umbrella | `include/layers/Layers.hpp` → aliases `nn::Lif`, `nn::LifBPTT`, `nn::LifIntegrator` |
| Generator | `cmake/GenerateLayers.cmake` — auto-builds `Layers.hpp` from filenames |
| LeakyReLU (NOT renamed) | `include/layers/activations/LeakyReLU.hpp` → `LeakyReLUImpl<Backend>` |

## Common pitfalls

1. **`LeakyReLU` is not a LIF layer.** Never rename `LeakyReLU` when renaming LIF layers. Use `(?!ReLU)` negative lookahead: `s/\bLeaky\b(?!ReLU)/Lif/g`.

2. **Python demo files use `snn.Leaky`** (SNNTorch library API). Do NOT rename those.

3. **Neuroscience prose stays "Leaky Integrate-and-Fire".** The biological concept name is "Leaky Integrate-and-Fire (LIF)". Only the C++ class is renamed. After mass rename, restore occurrences of "Lif Integrate-and-Fire" back to "Leaky Integrate-and-Fire".

4. **`Layers.hpp` is auto-generated** by `cmake/GenerateLayers.cmake` from filenames. After renaming a header file, re-run `cmake --preset=max-performance` to regenerate it. Do not hand-edit `Layers.hpp`.

5. **Three locations, always in sync.** Global (`~/.claude/commands/`) and project (`.claude/commands/`) Claude commands should be identical. OpenCode (`SKILL.md`) may have slightly different frontmatter format but same content.

## See also

- [agent-customization](./agent-customization.md) — create/validate new skill files
- `cmake/GenerateLayers.cmake` — layer alias auto-generation
- `include/layers/Layers.hpp` — canonical alias list
