---
name: experiment-reproducibility
description: "Enforce deterministic experiment metadata and output traceability."
---

# experiment-reproducibility

Goal
- Ensure every experiment run is reproducible and traceable.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: EXPLICIT_CONFIG
  DO: Persist run config in JSON/YAML artifacts.
  AVOID: Implicit defaults without serialization.
- RULE: RESULT_LINKAGE
  DO: Link each result to the exact profile/config used.
  AVOID: Detached outputs.
- RULE: TIMESTAMPED_OUTPUTS
  DO: Timestamp output folders/files.
  AVOID: Overwriting prior results.
- RULE: DETERMINISM
  DO: Record seeds and deterministic settings when applicable.
  AVOID: Non-repeatable runs.

Validation
- Run artifacts include config + timestamp + linkage metadata.
