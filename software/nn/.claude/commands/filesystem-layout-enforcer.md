---
description: "Enforce modular project layout for core libraries, experiments, profiles, and result artifacts."
---

# filesystem-layout-enforcer

Preserve repository modularity and discoverability. Every new file must land in the correct location.

## Project Context (nn framework)

**Canonical layout:**
```
include/layers/<category>/  — public layer headers (one per file)
src/core/                      — implementation + unit tests
src/experiments/<id>/          — experiment binary + lib/ + tests/ + profiles/
scripts/pipeline/              — paper generation chain
scripts/data/                  — dataset conversion
scripts/ci/                    — CI gate scripts (referenced by ci.yml)
scripts/dev/                   — developer tooling
results/                       — experiment output (CSV, JSON, NPY)
.wiki/                         — documentation wiki
```

**Scripts must land in the correct subdir** — `ci.yml` hardcodes paths to `scripts/ci/`. Moving scripts without updating the workflow breaks CI.

**No new files in `scripts/` root** — only `requirements.txt` and `README.md` belong there.

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

- **EXPERIMENT_LAYOUT**: Place reusable experiment code under `src/experiments/<id>/lib/include/` and `src/experiments/<id>/lib/src/`. No mixing reusable code into ad-hoc mains.
- **CORE_BOUNDARY**: Keep shared primitives in `src/core/` and public headers in `include/`. No duplicating core utilities inside experiments.
- **ARTIFACT_LAYOUT**: Keep profile and result artifacts in dedicated folders (`results/`, `profiles/`). No scattering generated outputs across source trees.

## Expected Layout

```
include/          ← public headers for core library
src/core/            ← core library implementations
src/experiments/
  <id>/
    main.cpp         ← entry point only
    lib/
      include/       ← experiment-local reusable headers
      src/           ← experiment-local reusable sources
results/             ← timestamped run outputs
profiles/            ← profiling artifacts
```

## Checklist

1. Is the new file a shared primitive? → `include/` + `src/core/`.
2. Is it experiment-specific and reusable within that experiment? → `src/experiments/<id>/lib/`.
3. Is it a run artifact (JSON, binary, log)? → `results/<experiment_id>/<timestamp>/`.
4. Does any new header duplicate something already in `include/`?

## Validation

- New files follow existing module boundaries.
- No generated artifacts committed to `src/` or `include/`.
