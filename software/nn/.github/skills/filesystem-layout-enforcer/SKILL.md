---
name: filesystem-layout-enforcer
description: "Enforce modular project layout for core libraries, experiments, profiles, and results."
---

# filesystem-layout-enforcer

Goal
- Preserve repository modularity and discoverability.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
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
