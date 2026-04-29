---
description: "Constrain model and pipeline choices to fit low-memory targets (Raspberry Pi class and similar)."
---

# memory-constrained-design

Keep experiments inside hard RAM budgets for embedded and edge targets.

## Rules

- **HARD_BUDGET**: Treat memory budget as a hard constraint, not a soft goal. Never optimize first and budget later.
- **SMALL_DEFAULTS**: Use `batch_size <= 8`, `latent <= 32`, `depth <= 2` unless explicitly justified otherwise.
- **DTYPE_DISCIPLINE**: Prefer `float32` for training/inference. No higher precision defaults on constrained devices.
- **TENSOR_CAP**: Keep single tensor allocations under practical limits (target: <10 MB). No silent large allocation spikes.

## Checklist

1. Check experiment config for budget-sensitive parameters (`batch_size`, `latent_dim`, `depth`, `time_steps`).
2. Estimate peak memory: sum of model weights + gradient buffers + activation tensors.
3. Flag any tensor allocation that could exceed 10 MB.
4. Confirm `dtype` is `float32` throughout.

## Validation

- Config declares budget-sensitive parameters explicitly.
- Memory-heavy operations are justified and measured.
- Peak memory estimate fits within declared hardware budget.
