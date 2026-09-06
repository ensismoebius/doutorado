---
name: nn-core-usage-enforcer
description: "Enforce reuse of existing nn core abstractions instead of reimplementation."
---

# nn-core-usage-enforcer

Goal
- Keep new work aligned with existing `nn` core contracts.

Rules
- RULE: CORE_REUSE
  DO: Use existing `Tensor`, `Layer`, `Sequential`, and core modules.
  AVOID: Reimplementing core abstractions.
- RULE: LAYER_REUSE
  DO: Reuse existing layers (`Linear`, `ReLU`, `Leaky*`) before adding new ones.
  AVOID: Duplicate forward/backward logic.
- RULE: API_COMPAT
  DO: Preserve core API semantics unless migration is explicitly requested.
  AVOID: Silent behavior drift.

Validation
- New code composes with existing core types.
- No duplicate tensor or training loop implementations.

Project Context (nn framework)
**Existing abstractions to reuse** (never reimplement):
- `Module<Backend>` — base for all layers (`include/layers/base/Module.hpp`)
- `Tensor` — all tensor ops (`include/tensor/Tensor.hpp`)
- `Adam`, `SGD` — optimizers (`include/optimizers/`)
- `KFold`, `StratifiedKFold`, `NestedKFold` — cross-validation (`include/statistics/kfold.hpp`)
- `NetworkSerializer` — save/load model (`include/serialization/NetworkSerializer.hpp`)
- `DataLoader`, `BatchPrefetcher` — data pipeline (`include/data_loaders/`)

**Anti-patterns:**
- Reimplementing matmul or normalization outside the `Tensor` interface → breaks backend abstraction
- Including `XTensorBackend.hpp` in `src/core/` targets → breaks portability
- Calling `clEnqueueWriteBuffer` directly instead of using `Tensor` → bypasses buffer pool
