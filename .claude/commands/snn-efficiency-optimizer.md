
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

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

Checklist

1. Audit temporal loop bounds — are `time_steps` justified by the experiment config?
2. Trace membrane/state buffer lifecycle — are they allocated inside the loop?
3. Check for full-tensor resets that could be incremental instead.
4. Run before/after smoke timing to confirm improvement.
