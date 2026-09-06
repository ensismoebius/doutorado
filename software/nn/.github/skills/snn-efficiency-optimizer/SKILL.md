---
name: snn-efficiency-optimizer
description: "Optimize SNN pipelines by reducing temporal recomputation and buffer churn."
---

# snn-efficiency-optimizer

Goal
- Minimize temporal overhead in spiking pipelines.

Rules
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

Project Context (nn framework)
**Membrane buffer lifecycle** — correct pattern:
- `v_mem_history` and `spike_history` pre-allocated in `LeakyBPTT::forward` **before** the time loop
- Re-allocation inside the loop = major regression; grep `v_mem_history` to verify

**Experiment baselines** (AMD Renoir APU, 5-fold CV):
- LSTM autoencoder: ~10 min per article profile run
- SNN autoencoder: ~45 min per article profile run (3 SNN profiles = ~2.25 h)
- Full article pipeline: ~2.5 h via `scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh`

**`time_steps` field** — set per profile in `src/experiments/guayaquil/profiles/*.json`:
```json
"model": { "time_steps": 10 }
```
Smaller `time_steps` = faster but may hurt accuracy. Article runs use profile-specified values.

**BatchScope** — outer `BatchScope` in `Trainer` (GPU builds only) batches all layer `clFinish` calls into one per mini-batch. Removing it restores 6× GPU stall overhead.
