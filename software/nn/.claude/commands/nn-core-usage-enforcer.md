---
description: "Enforce reuse of existing nn core abstractions (Tensor, Layer, Sequential) instead of reimplementation."
---

# nn-core-usage-enforcer

Keep new work aligned with existing `nn` core contracts.

## Project Context (nn framework)

**Existing abstractions to reuse** (never reimplement):
- `Module<Backend>` — base for all layers (`include/nn/layers/base/Module.hpp`)
- `Tensor` — all tensor ops (`include/nn/tensor/Tensor.hpp`)
- `Adam`, `SGD` — optimizers (`include/nn/optimizers/`)
- `KFold`, `StratifiedKFold`, `NestedKFold` — cross-validation (`include/nn/statistics/kfold.hpp`)
- `NetworkSerializer` — save/load model (`include/nn/saver/NetworkSerializer.hpp`)
- `DataLoader`, `BatchPrefetcher` — data pipeline (`include/nn/dataLoaders/`)

**Anti-patterns:**
- Reimplementing matmul or normalization outside the `Tensor` interface → breaks backend abstraction
- Including `XTensorBackend.hpp` in `src/core/` targets → breaks portability
- Calling `clEnqueueWriteBuffer` directly instead of using `Tensor` → bypasses buffer pool

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

## Rules

- **CORE_REUSE**: Use existing `Tensor`, `Layer`, `Sequential`, and core modules. No reimplementing core abstractions.
- **LAYER_REUSE**: Reuse existing layers (`Linear`, `ReLU`, `Leaky*`, etc.) before adding new ones. No duplicate forward/backward logic.
- **API_COMPAT**: Preserve core API semantics unless migration is explicitly requested. No silent behavior drift.

## Checklist (run before completing any task)

1. Search for the abstraction first: `rg "class <Candidate>" include/ src/`.
2. Check `include/nn/` for existing headers covering the need.
3. If a new class is truly needed, confirm it composes with existing core types.
4. Verify no duplicate tensor or training loop implementations are introduced.

## Validation

- New code composes with existing `Tensor`, `Layer`, and `Sequential` types.
- No duplicate tensor or training loop implementations.
- Build passes for all touched targets.
