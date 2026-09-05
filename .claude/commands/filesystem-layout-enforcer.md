
# filesystem-layout-enforcer

Goal
- Preserve repository modularity and discoverability. Every new file must land in the correct location.

Rules

- RULE: EXPERIMENT_LAYOUT
  DO: Place reusable experiment code under `src/experiments/<id>/lib/include/` and `src/experiments/<id>/lib/src/`
  AVOID: No mixing reusable code into ad-hoc mains
- RULE: CORE_BOUNDARY
  DO: Keep shared primitives in `src/core/` and public headers in `include/`
  AVOID: No duplicating core utilities inside experiments
- RULE: ARTIFACT_LAYOUT
  DO: Keep profile and result artifacts in dedicated folders (`results/`, `profiles/`)
  AVOID: No scattering generated outputs across source trees

Validation

- New files follow existing module boundaries.
- No generated artifacts committed to `src/` or `include/`.

Project Context (nn framework)

**Canonical layout:**
```
include/nn/layers/<category>/  — public layer headers (one per file)
src/core/                      — implementation + unit tests
src/experiments/<id>/          — experiment binary + lib/ + tests/ + profiles/
scripts/pipeline/              — paper generation chain
scripts/data/                  — dataset conversion
scripts/ci/                    — CI gate scripts (referenced by ci.yml)
scripts/dev/                   — developer tooling
results/                       — experiment output (CSV, JSON, NPY)
.wiki/                         — documentation wiki
```

**Scripts must land in the correct subdir** — `ci.yml` hardcodes paths to `scripts/ci/`. Moving scripts without updating the workflow breaks CI.

**No new files in `scripts/` root** — only `requirements.txt` and `README.md` belong there.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
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

Expected Layout

```
include/nn/          ← public headers for core library
src/core/            ← core library implementations
src/experiments/
  <id>/
    main.cpp         ← entry point only
    lib/
      include/       ← experiment-local reusable headers
      src/           ← experiment-local reusable sources
results/             ← timestamped run outputs
profiles/            ← profiling artifacts
```

Checklist

1. Is the new file a shared primitive? → `include/nn/` + `src/core/`.
2. Is it experiment-specific and reusable within that experiment? → `src/experiments/<id>/lib/`.
3. Is it a run artifact (JSON, binary, log)? → `results/<experiment_id>/<timestamp>/`.
4. Does any new header duplicate something already in `include/nn/`?
