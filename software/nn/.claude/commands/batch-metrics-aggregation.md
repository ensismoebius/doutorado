---
description: "Enforce a unified contract for batch-level metric accumulation (loss, accuracy, confusion matrix) across all training loops."
---

# batch-metrics-aggregation

Replace per-experiment ad-hoc metric collection with a standard accumulator that works consistently across all training loops and experiment types.

## Rules

- **ACCUMULATOR_PATTERN**: Use a `MetricAccumulator` object per epoch (not ad-hoc running sums). Reset it at the start of each epoch. No manual `total_loss += batch_loss / n_batches` scatter across loop bodies.
- **WEIGHTED_BY_BATCH_SIZE**: Accumulate loss weighted by batch size, not unweighted. The last batch may be smaller — unweighted averaging introduces bias. No `mean_loss = sum_loss / n_batches`.
- **VALIDITY_CHECK**: After each batch, assert that batch loss is finite and non-negative. Log `WARN` if loss increases by more than 10× between consecutive batches. No silently accumulating garbage.
- **UNIFIED_METRICS**: `EpochResult` must carry at minimum: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `epoch_duration_ms`. Extending it is fine; removing fields is not without updating all consumers.
- **CONFUSION_MATRIX_OPTIONAL**: When `compute_confusion_matrix: true` in config, the accumulator must collect predictions and targets for full confusion-matrix computation at epoch end. No recomputing them in a separate pass.
- **AUTOENCODER_MODE**: For autoencoder experiments, accumulate reconstruction loss only; accuracy is undefined and must not be reported as zero.

## Standard Accumulator Interface

```cpp
MetricAccumulator acc;
for (auto& [x, y] : train_loader) {
    auto loss = model.forward_loss(x, y);
    auto preds = model.predict(x);
    acc.update(loss.item(), preds, y, /*batch_size=*/x.rows());
}
EpochResult result = acc.finalize();
// result.train_loss  ← weighted mean loss
// result.train_acc   ← weighted mean accuracy
// result.confusion   ← optional confusion matrix
```

## Key Files to Fix

- [src/core/training/EpochResult.hpp](src/core/training/EpochResult.hpp) — add `train_acc`, `val_acc`, `epoch_duration_ms`
- [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — replace ad-hoc accumulators with `MetricAccumulator`
- [src/experiments/02/Experiment02Training.cpp](src/experiments/02/Experiment02Training.cpp) — uses different accumulation logic than Exp04

## Validation

- All experiments produce `EpochResult` with the same set of fields.
- Loss is weighted by batch size (last batch size logged for verification).
- `NaN` batch loss triggers `WARN`, not silent accumulation.
