---
description: "Enforce fold ID, split seed, and sample index tracking in all K-fold experiment outputs for post-hoc analysis."
---

# fold-metadata-tracker

Ensure K-fold results are never aggregated without first saving per-fold metadata, enabling fold-wise error analysis and significance testing.

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

**Fold results location:** `results/*_comparative_metrics.csv` — one row per (model, architecture, fold, epoch)

**KFold API:** `include/statistics/kfold.hpp`
- `KFold`, `StratifiedKFold`, `NestedKFold`

**CSV fold fields:**
- `fold` — 0-indexed fold number
- `run_id` — unique run identifier from profile `experiment.run_tag`
- `model` — `"lstm-ae"` or `"snn-ae"`
- `architecture` — `"dense"`, `"conv1d"`, or `"recurrent"`

## Rules

- **FOLD_ID_IN_OUTPUT**: Every per-fold result (loss, metrics, predictions) must be tagged with its fold ID (0-indexed). No aggregating fold results before saving per-fold records.
- **SPLIT_SEED_RECORDED**: The random seed used to generate the fold split must be saved alongside results. Two runs with the same seed must produce the same split.
- **SAMPLE_INDICES_SAVED**: Save train and validation sample indices per fold to a sidecar file. Enables re-running a single fold and verifying reproducibility.
- **NO_EARLY_AGGREGATION**: Compute and save per-fold metrics first, then aggregate to mean ± std. Never compute only the aggregate and discard fold-level data.
- **FOLD_TIMING**: Record wall-clock time per fold. Log summary (min, max, mean fold duration) at end of cross-validation.

## Required Output Structure

```
results/<experiment_id>/<timestamp>/
├── config.json
├── fold_00/
│   ├── metrics.json      ← { "fold_id": 0, "val_loss": ..., "val_acc": ... }
│   ├── indices.json      ← { "train": [...], "val": [...] }
│   └── model.bin         ← best checkpoint for this fold
├── fold_01/
│   └── ...
└── summary.json          ← { "mean_val_acc": ..., "std_val_acc": ..., "split_seed": 42 }
```

## Key Files to Fix

- [include/statistics/kfold.hpp](include/statistics/kfold.hpp) — `FoldSplit` struct needs `fold_id` and `split_seed` fields
- [src/experiments/waveletAE/WaveletAETraining.cpp](src/experiments/waveletAE/WaveletAETraining.cpp) — save per-fold metrics before averaging
- [src/core/training/EpochResult.hpp](src/core/training/EpochResult.hpp) — add `fold_id` field

## Validation

- `fold_<id>/metrics.json` exists for every fold after training completes.
- `fold_<id>/indices.json` allows exact reproduction of the train/val split.
- `summary.json` is written only after all per-fold files are flushed.
- Re-running with same seed and same fold ID produces the same split.
