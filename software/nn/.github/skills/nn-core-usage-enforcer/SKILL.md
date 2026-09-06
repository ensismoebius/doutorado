---
name: nn-core-usage-enforcer
description: "Enforce reuse of existing nn core abstractions (Tensor, Layer, Sequential) instead of reimplementation."
---

# nn-core-usage-enforcer

Keep new work aligned with existing `nn` core contracts.

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
