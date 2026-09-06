
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



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
  DO: Each profile JSON in `src/experiments/guayaquil/profiles/` must conform to a shared schema
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

**Sweep output:** `scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py` produces `results/paper_sweep_alpha.csv` and `results/paper_sweep_vth.csv` from the comparative metrics CSVs.

**Profile sweep fields:**
```json
"model": {
  "alpha_values": [0.5, 1.0, 2.0],
  "v_th_values": [0.5, 1.0, 1.5]
}
```
Each value in the array → one independent run with that hyperparameter fixed.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

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

- [src/experiments/guayaquil/profiles/](src/experiments/guayaquil/profiles/) — add `search_space.json` and schema
- [src/demos/pydemos/experiments/run_hyper_search.py](src/demos/pydemos/experiments/run_hyper_search.py) — add trial logging
- [src/demos/pydemos/experiments/run_targeted_search.py](src/demos/pydemos/experiments/run_targeted_search.py) — same
- [src/demos/pydemos/experiments/run_extensive_search.py](src/demos/pydemos/experiments/run_extensive_search.py) — same
