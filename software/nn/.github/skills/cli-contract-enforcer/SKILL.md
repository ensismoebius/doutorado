---
name: cli-contract-enforcer
description: "Enforce deterministic, side-effect-safe CLI behavior and argument validation."
---

# cli-contract-enforcer

Goal
- Keep CLI interfaces deterministic and safe.

Rules
- RULE: HELP_IS_PURE
  DO: Make `--help` exit immediately with status 0.
  AVOID: Side effects on help output.
- RULE: ARG_VALIDATION
  DO: Validate all required and range-constrained arguments.
  AVOID: Silent invalid-argument fallthrough.
- RULE: OVERRIDE_SUPPORT
  DO: Allow profile/config override flags explicitly.
  AVOID: Hidden precedence rules.

Validation
- Help path performs no file writes or side effects.
- Invalid inputs fail fast with clear diagnostics.

Project Context (nn framework)
**Main CLI entry points:**
- `experiment04 --comparative-config <profile.json>` — primary paper pipeline entry point
- `experiment04 --help` — must list all flags, exit 0, no side effects
- `experiment03 --help` — same requirement
- `Phase00 [config.json]` — optional positional arg for config path

**Profile JSON is the primary config surface** for Exp04. All training parameters (model, data, folds, hyperparams) come from the profile — no hidden defaults that differ between `--help` output and actual behavior.

**No side effects on `--help`:** `--help` must not write files, touch the filesystem, or open OpenCL devices. Use `nn::logging::StreamRedirector` only after help check.

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
