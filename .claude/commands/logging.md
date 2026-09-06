
# logging

Goal
- Centralize runtime output and remove ad-hoc console/file diagnostics.

Rules

- RULE: LOGGER_ONLY
  DO: Use `NN_LOG_ERROR` / `NN_LOG_WARN` / `NN_LOG_INFO` / `NN_LOG_DEBUG`
  AVOID: No new `std::cout`, `std::cerr`, or `/tmp/*.log` in core paths
- RULE: LEVEL_DISCIPLINE
  DO: Map severity correctly: `ERROR` for failures, `WARN` for degraded state, `INFO` for milestones, `DEBUG` for trace
  AVOID: No high-volume info logs in hot loops
- RULE: CAPTURE_CONSISTENCY
  DO: Use `StreamRedirector` where unified output is required
  AVOID: No mixed direct console and logger writes in the same path
- RULE: PERF_GATE
  DO: Keep logging out of tight inner loops unless gated by debug level
  AVOID: No high-frequency logging in hot paths

Workflow

1. Replace ad-hoc prints (`std::cout`, `std::cerr`, `printf`) in target files.
2. Add `#include "nn/logging/Logger.hpp"` where needed.
3. Build the affected target and run smoke path to validate output.

Validation

- No new ad-hoc prints in modified core files.
- Output remains readable with progress UI.
- Build passes after migration.

Project Context (nn framework)

**Logger header:** `include/nn/logging/Logger.hpp` — macros: `NN_LOG_ERROR`, `NN_LOG_WARN`, `NN_LOG_INFO`, `NN_LOG_DEBUG`

**Key log points in `Trainer.hpp`:**
- `INFO`: epoch start/end with loss values
- `DEBUG` (gated, not in hot path): per-batch loss when debug level enabled
- `ERROR`: NaN loss detected — log and abort training
- `WARN`: SNN biophysical param (R, C) hit clamp boundary

**Never log inside:**
- `LeakyBPTT` inner time loop — called `time_steps × batch_size` times per forward
- `matmul` inner K-loop — called `rows × cols × K` times
- Any loop with >1000 iterations in typical workload

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`
