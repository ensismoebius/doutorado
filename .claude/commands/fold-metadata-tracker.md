
# fold-metadata-tracker

Goal
- Ensure K-fold results are never aggregated without first saving per-fold metadata, enabling fold-wise error analysis and significance testing.

Rules

- RULE: FOLD_ID_IN_OUTPUT
  DO: Every per-fold result (loss, metrics, predictions) must be tagged with its fold ID (0-indexed)
  AVOID: No aggregating fold results before saving per-fold records
- RULE: SPLIT_SEED_RECORDED
  DO: The random seed used to generate the fold split must be saved alongside results. Two runs with the same seed must produce the same split
- RULE: SAMPLE_INDICES_SAVED
  DO: Save train and validation sample indices per fold to a sidecar file. Enables re-running a single fold and verifying reproducibility
- RULE: NO_EARLY_AGGREGATION
  DO: Compute and save per-fold metrics first, then aggregate to mean ± std
  AVOID: Never compute only the aggregate and discard fold-level data
- RULE: FOLD_TIMING
  DO: Record wall-clock time per fold. Log summary (min, max, mean fold duration) at end of cross-validation

Validation

- `fold_<id>/metrics.json` exists for every fold after training completes.
- `fold_<id>/indices.json` allows exact reproduction of the train/val split.
- `summary.json` is written only after all per-fold files are flushed.
- Re-running with same seed and same fold ID produces the same split.

Project Context (nn framework)

**Fold results location:** `results/*_comparative_metrics.csv` — one row per (model, architecture, fold, epoch)

**KFold API:** `include/nn/statistics/kfold.hpp`
- `KFold`, `StratifiedKFold`, `NestedKFold`

**CSV fold fields:**
- `fold` — 0-indexed fold number
- `run_id` — unique run identifier from profile `experiment.run_tag`
- `model` — `"lstm-ae"` or `"snn-ae"`
- `architecture` — `"dense"`, `"conv1d"`, or `"recurrent"`

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

Required Output Structure

```
results/<experiment_id>/<timestamp>/
├── config.json
├── fold_00/
│   ├── metrics.json      ← { "fold_id": 0, "val_loss": ..., "val_acc": ... }
│   ├── indices.json      ← { "train": [...], "val": [...] }
│   └── model.bin         ← best checkpoint for this fold
├── fold_01/
│   └── ...
└── summary.json          ← { "mean_val_acc": ..., "std_val_acc": ..., "split_seed": 42 }
```

Key Files to Fix

- [include/nn/statistics/kfold.hpp](include/nn/statistics/kfold.hpp) — `FoldSplit` struct needs `fold_id` and `split_seed` fields
- [src/experiments/waveletAE/WaveletAETraining.cpp](src/experiments/waveletAE/WaveletAETraining.cpp) — save per-fold metrics before averaging
- [src/core/training/EpochResult.hpp](src/core/training/EpochResult.hpp) — add `fold_id` field
