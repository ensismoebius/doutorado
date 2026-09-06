---
name: data-normalization-contract-enforcer
description: "Enforce normalization scope (per-feature vs global), train-only fitting, and leakage prevention between splits."
---

# data-normalization-contract-enforcer

Ensure that normalization is applied exactly as declared in experiment config — especially that statistics are never computed on validation or test data.

## Rules

- **FIT_TRAIN_ONLY**: Normalization statistics (mean, std, min, max) must be computed exclusively on the training fold. No fitting on validation or test subsets, even partially.
- **SCOPE_HONORED**: If config declares `scope: "per_feature"`, normalize each feature independently. If `scope: "global"`, normalize across all features. No mismatch between config and implementation.
- **STATS_LOGGED**: Log computed normalization statistics (per-feature mean/std or global min/max) at `INFO` level before applying. No unrecorded normalization transforms.
- **APPLY_CONSISTENT**: Apply the same statistics (fitted on train) to both train and validation sets. No re-computing stats on validation.
- **LEAKAGE_CHECK**: Before training, assert that val/test split indices were excluded from the stats computation. Log a warning if the split is not verified.
- **MODALITY_CONSISTENT**: When multiple modalities are normalized (EEG, audio), ensure each modality uses its declared normalization method. No silently applying z-score to a min-max-specified modality.

## Key Files to Audit

- [include/utility/Normalization.hpp](include/utility/Normalization.hpp) — `normalize_0_1()`: check it accepts pre-computed stats
- [include/utility/EEGWindowZScore.hpp](include/utility/EEGWindowZScore.hpp) — z-score contract: fit separately per fold?
- [include/utility/AudioMeanStdNormalize.hpp](include/utility/AudioMeanStdNormalize.hpp) — mean-std contract
- [src/experiments/waveletAE/spec.yaml](src/experiments/waveletAE/spec.yaml) — `normalization: {method, scope, fit_on}` spec

## Correct Pattern

```cpp
// Fit on train indices only:
auto stats = compute_stats(dataset, train_indices);  // NOT all indices
NN_LOG_INFO("Norm stats: mean={}, std={}", stats.mean, stats.std);

// Apply same stats to both splits:
auto train_norm = apply_norm(dataset, train_indices, stats);
auto val_norm   = apply_norm(dataset, val_indices,   stats);  // same stats
```

## Validation

- Stats computation function receives only train indices.
- Validation data normalized with train-fitted stats (not re-fitted).
- Normalization method matches `spec.yaml` declaration for each modality.

Project Context (nn framework)
**Normalization sites in Exp04:**
- **z-score per window:** `src/experiments/guayaquil/lib/src/ComparativeDataset.cpp:37` — applied to raw EEG/audio windows; statistics computed on training fold only ✓
- **per-encoding min/max re-normalization:** `src/experiments/guayaquil/lib/src/ComparativeEncoding.cpp` — applied after input transform (dense/conv1d/recurrent); leakage risk here if stats are computed over full dataset

**Leakage risk:** The encoding step in `ComparativeEncoding.cpp` must refit normalization stats on the training fold only. If the encoding transform changes the data distribution, verify that the re-normalization step uses only train-fold statistics.
