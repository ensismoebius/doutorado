---
name: experiment-reproducibility
description: "Enforce deterministic experiment metadata, config serialization, and output traceability."
---

# experiment-reproducibility

Ensure every experiment run is reproducible and traceable.

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

**Article pipeline chain** (full paper reproduction):
```
scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh
  → results/article_*_comparative_metrics.csv
  → scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py
  → documentation/.../data/article_*_*.dat
  → pdflatex paper.tex
```

**Profile locations:** `src/experiments/guayaquil/profiles/article-{lstm-ae,snn-dense,snn-conv1d,snn-recurrent}.json`

**Profile audit** — run after any profile edit:
```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

**`seed_deterministic` field** — set `false` in all article profiles (random init). Set `true` only for debugging/unit tests. Value recorded in CSV output for traceability.

## Rules

- **EXPLICIT_CONFIG**: Persist run config in JSON/YAML artifacts alongside results. No implicit defaults without serialization.
- **RESULT_LINKAGE**: Link each result to the exact profile/config used. No detached outputs.
- **TIMESTAMPED_OUTPUTS**: Timestamp output folders/files (ISO 8601: `YYYYMMDD_HHMMSS`). No overwriting prior results.
- **DETERMINISM**: Record seeds and deterministic settings (`torch.manual_seed`, OpenCL queue order, etc.). No non-repeatable runs.

## Required Artifact Structure

Each run must produce:
```
results/<experiment_id>/<YYYYMMDD_HHMMSS>/
├── config.json      ← full run configuration (hyperparams, paths, seeds)
├── metrics.json     ← quantitative results
└── <model>.bin      ← saved model state (optional but preferred)
```

## Validation

- Run artifacts include config + timestamp + result linkage metadata.
- Config file is human-readable JSON or YAML.
- No two runs share the same output directory.
