
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# navigation

Goal
- Find the right file/symbol fast, with reproducible search steps.

Rules

- RULE: SEARCH_NARROWLY
  DO: Start in `src/`, `include/`, `cmake/`, `scripts/`
  AVOID: Avoid `build/`, `_deps/`, generated outputs unless requested
- RULE: CHEAP_TO_DEEP
  DO: Use the code-intelligence MCP (symbol lookup → structure → source range) before `rg`, and `rg`/file-read only for what it doesn't cover
  AVOID: Never read large files before locating symbol anchors
- RULE: USAGE_CONFIRMATION
  DO: Use `find_references`/`find_dependencies` for cross-file impact, and read the `"exact"`/`"heuristic"` tag on every row before trusting it
  AVOID: Never assume from a single occurrence; never read a heuristic "0 callers" as dead code
- RULE: PERF_GATE
  DO: When locating hot paths, include allocation and loop hotspots explicitly
  AVOID: Don't produce search plans that ignore memory/CPU cost centers

Workflow

1. Symbol or concept name known → `find_symbol` (MCP). File known, need its
   contents → `get_file_structure` (symbols, no source) then
   `get_source_range`/`symbol_source` for the exact lines.
2. Neither known → `search_text` (MCP): a regex/text search scoped to
   indexed files, `src/`/`include/`/`cmake/`/`scripts/` only (`build/`,
   `_deps/`, generated output are excluded automatically — no manual
   `-path`/`! -path` filtering needed), each hit tagged with its enclosing
   symbol. Fall back to `rg`/`find` only for something outside the index
   (a config file, a script, a build artifact) or if the MCP is unavailable.
3. Read only the needed line ranges (`get_source_range` already budgets
   this — an oversized request reports `{"truncated": true,
   "recommended_ranges": [...]}` instead of a silent clip).
4. Confirm references with `find_references`/`find_dependencies`, across
   all usages and related tests — not a single grep hit.

Validation

- Candidate list includes implementation + header + tests.
- Search scope excludes irrelevant directories (`build/`, `_deps/`) — automatic via the MCP, explicit `!` filters when falling back to `find`/`rg`.

Project Context (nn framework)

**Key file index:**

| What | Where |
|---|---|
| Module base | `include/nn/layers/base/Module.hpp` |
| LIF single-step | `include/nn/layers/spiking/Leaky.hpp` |
| LIF BPTT | `include/nn/layers/spiking/LeakyBPTT.hpp` |
| Trainer | `src/core/training/Trainer.hpp` |
| Exp04 profile parser | `src/experiments/guayaquil/lib/include/ComparativeConfig.hpp` |
| Exp04 encoding transforms | `src/experiments/guayaquil/lib/src/ComparativeEncoding.cpp` |
| KFold / NestedKFold | `include/nn/statistics/kfold.hpp` |
| OpenCL context | `include/nn/tensor/opencl/OpenCLContext.hpp` |

**SNN search anchors** — `search_text` (MCP) or grep for these to find the relevant code:
- `v_mem_history`, `spike_history` — membrane and spike recording in LeakyBPTT
- `time_steps`, `surrogate_grad`, `readout_mode` — SNN config fields
- `R_min`, `C_min` — biophysical parameter clamp sites

**Experiment paths:**
- `src/experiments/paraconsistentBaseline/` — wavelet + paraconsistent baseline
- `src/experiments/waveletAE/` — wavelet autoencoder
- `src/experiments/autoencoderRunner/` — multimodal autoencoder
- `src/experiments/guayaquil/` — LSTM vs SNN comparative

**Article pipeline chain:**
`scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh` → CSVs → `scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py` → DAT → `pdflatex paper.tex`

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`
