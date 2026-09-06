---
name: fold-metadata-tracker
description: "Enforce fold ID, split seed, and sample index tracking in all K-fold experiment outputs for post-hoc analysis."
---

# fold-metadata-tracker

Goal
- Ensure K-fold results are never aggregated without first saving per-fold metadata, enabling fold-wise error analysis and significance testing.

Rules

- RULE: FOLD_ID_IN_OUTPUT
  DO: Every per-fold result (loss, metrics, predictions) must be tagged with its fold ID (0-indexed)
  AVOID: No aggregating fold results before saving per-fold records
- RULE: SPLIT_SEED_RECORDED
  DO: The random seed used to generate the fold split must be saved alongside results. Two runs with the same seed must produce the same split
- RULE: SAMPLE_INDICES_SAVED
  DO: Save train and validation sample indices per fold to a sidecar file. Enables re-running a single fold and verifying reproducibility
- RULE: NO_EARLY_AGGREGATION
  DO: Compute and save per-fold metrics first, then aggregate to mean ± std
  AVOID: Never compute only the aggregate and discard fold-level data
- RULE: FOLD_TIMING
  DO: Record wall-clock time per fold. Log summary (min, max, mean fold duration) at end of cross-validation

Validation

- `fold_<id>/metrics.json` exists for every fold after training completes.
- `fold_<id>/indices.json` allows exact reproduction of the train/val split.
- `summary.json` is written only after all per-fold files are flushed.
- Re-running with same seed and same fold ID produces the same split.

Project Context (nn framework)

**Fold results location:** `results/*_comparative_metrics.csv` — one row per (model, architecture, fold, epoch)

**KFold API:** `include/nn/statistics/kfold.hpp`
- `KFold`, `StratifiedKFold`, `NestedKFold`

**CSV fold fields:**
- `fold` — 0-indexed fold number
- `run_id` — unique run identifier from profile `experiment.run_tag`
- `model` — `"lstm-ae"` or `"snn-ae"`
- `architecture` — `"dense"`, `"conv1d"`, or `"recurrent"`

Required Output Structure

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

Key Files to Fix

- [include/nn/statistics/kfold.hpp](include/nn/statistics/kfold.hpp) — `FoldSplit` struct needs `fold_id` and `split_seed` fields
- [src/experiments/waveletAE/WaveletAETraining.cpp](src/experiments/waveletAE/WaveletAETraining.cpp) — save per-fold metrics before averaging
- [src/core/training/EpochResult.hpp](src/core/training/EpochResult.hpp) — add `fold_id` field
