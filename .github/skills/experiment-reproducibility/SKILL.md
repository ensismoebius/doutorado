---
name: experiment-reproducibility
description: "Enforce deterministic experiment metadata, config serialization, and output traceability."
---

## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# experiment-reproducibility

Goal
- Ensure every experiment run is reproducible and traceable.

Rules

- RULE: EXPLICIT_CONFIG
  DO: Persist run config in JSON/YAML artifacts alongside results
  AVOID: No implicit defaults without serialization
- RULE: RESULT_LINKAGE
  DO: Link each result to the exact profile/config used
  AVOID: No detached outputs
- RULE: TIMESTAMPED_OUTPUTS
  DO: Timestamp output folders/files (ISO 8601: `YYYYMMDD_HHMMSS`)
  AVOID: No overwriting prior results
- RULE: DETERMINISM
  DO: Record seeds and deterministic settings (`torch.manual_seed`, OpenCL queue order, etc.)
  AVOID: No non-repeatable runs

Validation

- Run artifacts include config + timestamp + result linkage metadata.
- Config file is human-readable JSON or YAML.
- No two runs share the same output directory.

Project Context (nn framework)

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

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Required Artifact Structure

Each run must produce:
```
results/<experiment_id>/<YYYYMMDD_HHMMSS>/
├── config.json      ← full run configuration (hyperparams, paths, seeds)
├── metrics.json     ← quantitative results
└── <model>.bin      ← saved model state (optional but preferred)
```
