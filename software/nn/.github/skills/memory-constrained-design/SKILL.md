---
name: memory-constrained-design
description: "Constrain model and pipeline choices for low-memory targets (e.g., Raspberry Pi class devices)."
---

# memory-constrained-design

Goal
- Keep experiments inside hard RAM budgets.

Rules
- RULE: NO_LEGACY_FORWARDERS
  DO: Migrate call sites to canonical headers and update include paths directly.
  AVOID: Creating legacy forwarding headers, compatibility wrapper headers, or shim include files.
- RULE: HARD_BUDGET
  DO: Treat memory budget as a hard constraint.
  AVOID: Optimizing first and budgeting later.
- RULE: SMALL_DEFAULTS
  DO: Use `batch_size <= 8`, `latent <= 32`, `depth <= 2` unless justified.
  AVOID: Large default model shapes.
- RULE: DTYPE_DISCIPLINE
  DO: Prefer `float32` for training/inference unless required otherwise.
  AVOID: Higher precision defaults on constrained devices.
- RULE: TENSOR_CAP
  DO: Keep single tensor allocations under practical limits (target: <10MB).
  AVOID: Silent large allocation spikes.

Validation
- Config declares budget-sensitive parameters explicitly.
- Memory-heavy operations are justified and measured.
