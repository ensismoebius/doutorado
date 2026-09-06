---
name: filesystem-layout-enforcer
description: "Enforce modular project layout for core libraries, experiments, profiles, and results."
---

# filesystem-layout-enforcer

Goal
- Preserve repository modularity and discoverability.

Rules
- RULE: EXPERIMENT_LAYOUT
  DO: Place reusable experiment code under `src/experiments/<id>/lib/include` and `src/experiments/<id>/lib/src`.
  AVOID: Mixing reusable code into ad-hoc mains.
- RULE: CORE_BOUNDARY
  DO: Keep shared primitives in `src/core` and public headers in `include/`.
  AVOID: Duplicating core utilities inside experiments.
- RULE: ARTIFACT_LAYOUT
  DO: Keep profile and result artifacts in dedicated folders.
  AVOID: Scattering generated outputs across source trees.

Validation
- New files follow existing module boundaries.

Project Context (nn framework)
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
