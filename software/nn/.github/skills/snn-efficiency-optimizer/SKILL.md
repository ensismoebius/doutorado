---
name: snn-efficiency-optimizer
description: "Optimize SNN pipelines by reducing temporal recomputation and buffer churn."
---

# snn-efficiency-optimizer

Goal
- Minimize temporal overhead in spiking pipelines.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: TEMPORAL_COST_CONTROL
  DO: Keep `time_steps` as small as correctness allows.
  AVOID: Unbounded temporal loops by default.
- RULE: BUFFER_REUSE
  DO: Reuse membrane/state buffers across timesteps.
  AVOID: Per-step reallocations.
- RULE: INCREMENTAL_UPDATES
  DO: Prefer event-driven/incremental updates when architecture permits.
  AVOID: Full state recomputation every timestep.

Validation
- No unnecessary per-timestep allocations.
- Temporal loop changes preserve model semantics.
