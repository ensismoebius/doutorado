# statistics

Purpose
- Statistical utilities and metrics (confusion matrices, multi-class metrics, and helpers used by evaluations).
- Deterministic cross-validation splitters (`KFold` and `StratifiedKFold`) for reusable train/validation index generation.

Usage
- Include headers from `include/nn/statistics` and call provided functions on prediction/label arrays.
- For fold generation, include `nn/statistics/kfold.hpp`.

K-Fold API
- `statistics::KFold(n_splits, shuffle, random_seed)`:
	creates deterministic fold train/test index splits over sample indices.
- `statistics::StratifiedKFold(n_splits, shuffle, random_seed)`:
	creates deterministic fold splits while balancing class labels.
- Legacy `statistics::k_fold_cross_validation(...)` remains available and now delegates fold creation to `KFold` internally.

Tests
- See unit tests under `src/core/statistics/tests/` for usage and expected outputs.

Recent behavior notes
- `variance(...)` now validates inputs and throws on empty data (or zero-length raw arrays).
- `compute_classification_metrics(...)` now throws `std::invalid_argument` for mismatched label vector sizes and `std::runtime_error` for empty label vectors.
