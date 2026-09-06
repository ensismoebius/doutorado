
# nn-core-usage-enforcer

Goal
- Keep new work aligned with existing `nn` core contracts.

Rules

- RULE: CORE_REUSE
  DO: Use existing `Tensor`, `Layer`, `Sequential`, and core modules
  AVOID: No reimplementing core abstractions
- RULE: LAYER_REUSE
  DO: Reuse existing layers (`Linear`, `ReLU`, `Leaky*`, etc.) before adding new ones
  AVOID: No duplicate forward/backward logic
- RULE: API_COMPAT
  DO: Preserve core API semantics unless migration is explicitly requested
  AVOID: No silent behavior drift

Validation

- New code composes with existing `Tensor`, `Layer`, and `Sequential` types.
- No duplicate tensor or training loop implementations.
- Build passes for all touched targets.

Project Context (nn framework)

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

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

**Wiki & knowledge graph** (concepts, papers, docs — not code symbols; use the MCP tools above for those):
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

Checklist (run before completing any task)

1. Search for the abstraction first: `rg "class <Candidate>" include/ src/`.
2. Check `include/nn/` for existing headers covering the need.
3. If a new class is truly needed, confirm it composes with existing core types.
4. Verify no duplicate tensor or training loop implementations are introduced.
