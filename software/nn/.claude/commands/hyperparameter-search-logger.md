---
description: "Enforce hyperparameter search space declaration, sampled-point logging, and result-to-config traceability."
---

# hyperparameter-search-logger

Ensure every hyperparameter search is reproducible: the search space, every sampled point, and the winning config are all recorded together.

## Rules

- **SPACE_DECLARED**: The hyperparameter search space must be declared as a JSON file (`search_space.json`) before search begins. No ad-hoc profile variants without a declared space.
- **POINT_LOGGED**: Every evaluated hyperparameter combination must be logged to a results file with: config snapshot, final metric (e.g., val_loss, val_acc), and training duration. No lost trial data.
- **WINNER_LINKED**: The best-performing config must be explicitly copied or linked as `best_config.json` in the search output directory. No identifying the winner only from logs.
- **SEARCH_SEED_RECORDED**: Random search must record its seed. Grid search must record the grid coordinates. No non-reproducible search runs.
- **PROFILE_SCHEMA**: Each profile JSON in `src/experiments/04/profiles/` must conform to a shared schema. No ad-hoc key additions without schema update.
- **PYTHON_C_BRIDGE**: When Python scripts drive search over C++ executables (as in `pydemos/`), the Python script must serialize the full config to JSON and pass it as a file path to the binary — not as CLI flags. No hidden parameter passing.

## Required Search Output Structure

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

## Key Files

- [src/experiments/04/profiles/](src/experiments/04/profiles/) — add `search_space.json` and schema
- [src/demos/pydemos/experiments/run_hyper_search.py](src/demos/pydemos/experiments/run_hyper_search.py) — add trial logging
- [src/demos/pydemos/experiments/run_targeted_search.py](src/demos/pydemos/experiments/run_targeted_search.py) — same
- [src/demos/pydemos/experiments/run_extensive_search.py](src/demos/pydemos/experiments/run_extensive_search.py) — same

## Validation

- `search_space.json` exists and is valid JSON before search runs.
- One `trial_<id>.json` file per evaluated configuration.
- `best_config.json` exists and its metrics match the best trial in `trials/`.
- Re-running with same seed and same search space produces the same trial order.
