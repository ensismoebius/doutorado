---
name: experiment-reproducibility
description: "Enforce deterministic experiment metadata, config serialization, and output traceability."
---

# experiment-reproducibility

Goal
- Ensure every experiment run is reproducible and traceable.

Rules

- RULE: EXPLICIT_CONFIG
  DO: Persist run config in JSON/YAML artifacts alongside results
  AVOID: No implicit defaults without serialization
- RULE: RESULT_LINKAGE
  DO: Link each result to the exact profile/config used
  AVOID: No detached outputs
- RULE: TIMESTAMPED_OUTPUTS
  DO: Timestamp output folders/files (ISO 8601: `YYYYMMDD_HHMMSS`)
  AVOID: No overwriting prior results
- RULE: DETERMINISM
  DO: Record seeds and deterministic settings (`torch.manual_seed`, OpenCL queue order, etc.)
  AVOID: No non-repeatable runs

Validation

- Run artifacts include config + timestamp + result linkage metadata.
- Config file is human-readable JSON or YAML.
- No two runs share the same output directory.

Project Context (nn framework)

**Article pipeline chain** (full paper reproduction):
```
scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh
  → results/article_*_comparative_metrics.csv
  → scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py
  → documentation/.../data/article_*_*.dat
  → pdflatex paper.tex
```

**Profile locations:** `src/experiments/guayaquil/profiles/article-{lstm-ae,snn-dense,snn-conv1d,snn-recurrent}.json`

**Profile audit** — run after any profile edit:
```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

**`seed_deterministic` field** — set `false` in all article profiles (random init). Set `true` only for debugging/unit tests. Value recorded in CSV output for traceability.

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

Required Artifact Structure

Each run must produce:
```
results/<experiment_id>/<YYYYMMDD_HHMMSS>/
├── config.json      ← full run configuration (hyperparams, paths, seeds)
├── metrics.json     ← quantitative results
└── <model>.bin      ← saved model state (optional but preferred)
```
