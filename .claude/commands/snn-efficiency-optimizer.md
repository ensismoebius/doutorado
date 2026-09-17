
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# snn-efficiency-optimizer

Goal
- Minimize temporal overhead in spiking neural network pipelines.

Rules

- RULE: TEMPORAL_COST_CONTROL
  DO: Keep `time_steps` as small as correctness allows
  AVOID: No unbounded temporal loops by default
- RULE: BUFFER_REUSE
  DO: Reuse membrane/state buffers across timesteps
  AVOID: No per-step reallocations
- RULE: INCREMENTAL_UPDATES
  DO: Prefer event-driven/incremental updates when architecture permits
  AVOID: No full state recomputation every timestep

Validation

- No unnecessary per-timestep allocations.
- Temporal loop changes preserve model semantics and output determinism.
- Confirmed with `build-test` before reporting complete.

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

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Checklist

1. Audit temporal loop bounds — are `time_steps` justified by the experiment config?
2. Trace membrane/state buffer lifecycle — are they allocated inside the loop?
3. Check for full-tensor resets that could be incremental instead.
4. Run before/after smoke timing to confirm improvement.
