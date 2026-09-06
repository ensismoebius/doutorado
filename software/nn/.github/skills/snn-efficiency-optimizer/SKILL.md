---
name: snn-efficiency-optimizer
description: "Optimize SNN pipelines by reducing temporal recomputation and membrane buffer churn."
---

# snn-efficiency-optimizer

Minimize temporal overhead in spiking neural network pipelines.

## Code intelligence (MCP `code_intelligence`)

Prefer over grep/manual git/cmake for anything about the code itself:
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` / `replace_symbol` — structural findings, complexity hotspots, and hash-gated multi-site renames/edits
- `run_build` / `run_tests` / `run_lint` / `run_format` — structured build/test/lint output, not raw logs
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

## Project Context (nn framework)

**Membrane buffer lifecycle** — correct pattern:
- `v_mem_history` and `spike_history` pre-allocated in `LifBPTT::forward` **before** the time loop
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
