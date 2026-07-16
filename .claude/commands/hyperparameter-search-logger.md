
# hyperparameter-search-logger

Goal
- Ensure every hyperparameter search is reproducible: the search space, every sampled point, and the winning config are all recorded together.

Rules

- RULE: SPACE_DECLARED
  DO: The hyperparameter search space must be declared as a JSON file (`search_space.json`) before search begins
  AVOID: No ad-hoc profile variants without a declared space
- RULE: POINT_LOGGED
  DO: Every evaluated hyperparameter combination must be logged to a results file with: config snapshot, final metric (e.g., val_loss, val_acc), and training duration
  AVOID: No lost trial data
- RULE: WINNER_LINKED
  DO: The best-performing config must be explicitly copied or linked as `best_config.json` in the search output directory
  AVOID: No identifying the winner only from logs
- RULE: SEARCH_SEED_RECORDED
  DO: Random search must record its seed. Grid search must record the grid coordinates
  AVOID: No non-reproducible search runs
- RULE: PROFILE_SCHEMA
  DO: Each profile JSON in `src/experiments/04/profiles/` must conform to a shared schema
  AVOID: No ad-hoc key additions without schema update
- RULE: PYTHON_C_BRIDGE
  DO: When Python scripts drive search over C++ executables (as in `pydemos/`), the Python script must serialize the full config to JSON and pass it as a file path to the binary — not as CLI flags
  AVOID: No hidden parameter passing

Validation

- `search_space.json` exists and is valid JSON before search runs.
- One `trial_<id>.json` file per evaluated configuration.
- `best_config.json` exists and its metrics match the best trial in `trials/`.
- Re-running with same seed and same search space produces the same trial order.

Project Context (nn framework)

**Sweep parameters in Exp04 article profiles:**
- `article-snn-recurrent.json`: alpha sweep (surrogate gradient width)
- `article-snn-dense.json`: v_th sweep (spike threshold)
- Sweep range declared as a JSON array in the profile field

**Sweep output:** `scripts/pipeline/e04/02_e04_build_lstm_vs_snn_paper_data.py` produces `results/paper_sweep_alpha.csv` and `results/paper_sweep_vth.csv` from the comparative metrics CSVs.

**Profile sweep fields:**
```json
"model": {
  "alpha_values": [0.5, 1.0, 2.0],
  "v_th_values": [0.5, 1.0, 1.5]
}
```
Each value in the array → one independent run with that hyperparameter fixed.

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

Required Search Output Structure

```
search/<experiment_id>/<timestamp>/
├── search_space.json       ← declared parameter ranges/distributions
├── search_seed.txt         ← seed used for random/Bayesian sampling
├── trials/
│   ├── trial_000.json      ← { "config": {...}, "val_loss": ..., "duration_s": ... }
│   ├── trial_001.json
│   └── ...
└── best_config.json        ← copy of winning trial config
```

Key Files

- [src/experiments/04/profiles/](src/experiments/04/profiles/) — add `search_space.json` and schema
- [src/demos/pydemos/experiments/run_hyper_search.py](src/demos/pydemos/experiments/run_hyper_search.py) — add trial logging
- [src/demos/pydemos/experiments/run_targeted_search.py](src/demos/pydemos/experiments/run_targeted_search.py) — same
- [src/demos/pydemos/experiments/run_extensive_search.py](src/demos/pydemos/experiments/run_extensive_search.py) — same
