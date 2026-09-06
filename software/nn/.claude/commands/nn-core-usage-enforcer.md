---
description: "Enforce reuse of existing nn core abstractions (Tensor, Layer, Sequential) instead of reimplementation."
---

# nn-core-usage-enforcer

Keep new work aligned with existing `nn` core contracts.

## Project Context (nn framework)

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

## Rules

- **CORE_REUSE**: Use existing `Tensor`, `Layer`, `Sequential`, and core modules. No reimplementing core abstractions.
- **LAYER_REUSE**: Reuse existing layers (`Linear`, `ReLU`, `Lif*`, etc.) before adding new ones. No duplicate forward/backward logic.
- **API_COMPAT**: Preserve core API semantics unless migration is explicitly requested. No silent behavior drift.

## Checklist (run before completing any task)

1. Search for the abstraction first: `rg "class <Candidate>" include/ src/`.
2. Check `include/` for existing headers covering the need.
3. If a new class is truly needed, confirm it composes with existing core types.
4. Verify no duplicate tensor or training loop implementations are introduced.

## Validation

- New code composes with existing `Tensor`, `Layer`, and `Sequential` types.
- No duplicate tensor or training loop implementations.
- Build passes for all touched targets.
