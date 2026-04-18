---
name: nn-core-usage-enforcer
description: "Enforce reuse of existing nn core abstractions instead of reimplementation."
---

# nn-core-usage-enforcer

Goal
- Keep new work aligned with existing `nn` core contracts.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: CORE_REUSE
  DO: Use existing `Tensor`, `Layer`, `Sequential`, and core modules.
  AVOID: Reimplementing core abstractions.
- RULE: LAYER_REUSE
  DO: Reuse existing layers (`Linear`, `ReLU`, `Leaky*`) before adding new ones.
  AVOID: Duplicate forward/backward logic.
- RULE: API_COMPAT
  DO: Preserve core API semantics unless migration is explicitly requested.
  AVOID: Silent behavior drift.

Validation
- New code composes with existing core types.
- No duplicate tensor or training loop implementations.
