---
description: "Optimize SNN pipelines by reducing temporal recomputation and membrane buffer churn."
---

# snn-efficiency-optimizer

Minimize temporal overhead in spiking neural network pipelines.

## Rules

- **TEMPORAL_COST_CONTROL**: Keep `time_steps` as small as correctness allows. No unbounded temporal loops by default.
- **BUFFER_REUSE**: Reuse membrane/state buffers across timesteps. No per-step reallocations.
- **INCREMENTAL_UPDATES**: Prefer event-driven/incremental updates when architecture permits. No full state recomputation every timestep.

## Checklist

1. Audit temporal loop bounds — are `time_steps` justified by the experiment config?
2. Trace membrane/state buffer lifecycle — are they allocated inside the loop?
3. Check for full-tensor resets that could be incremental instead.
4. Run before/after smoke timing to confirm improvement.

## Validation

- No unnecessary per-timestep allocations.
- Temporal loop changes preserve model semantics and output determinism.
- Confirmed with `build-test` before reporting complete.
