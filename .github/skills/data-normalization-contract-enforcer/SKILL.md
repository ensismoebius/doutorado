---
name: data-normalization-contract-enforcer
description: "Enforce normalization scope (per-feature vs global), train-only fitting, and leakage prevention between splits."
---

## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# data-normalization-contract-enforcer

Goal
- Ensure that normalization is applied exactly as declared in experiment config — especially that statistics are never computed on validation or test data.

Rules

- RULE: FIT_TRAIN_ONLY
  DO: Normalization statistics (mean, std, min, max) must be computed exclusively on the training fold
  AVOID: No fitting on validation or test subsets, even partially
- RULE: SCOPE_HONORED
  DO: If config declares `scope: "per_feature"`, normalize each feature independently. If `scope: "global"`, normalize across all features
  AVOID: No mismatch between config and implementation
- RULE: STATS_LOGGED
  DO: Log computed normalization statistics (per-feature mean/std or global min/max) at `INFO` level before applying
  AVOID: No unrecorded normalization transforms
- RULE: APPLY_CONSISTENT
  DO: Apply the same statistics (fitted on train) to both train and validation sets
  AVOID: No re-computing stats on validation
- RULE: LEAKAGE_CHECK
  DO: Before training, assert that val/test split indices were excluded from the stats computation. Log a warning if the split is not verified
- RULE: MODALITY_CONSISTENT
  DO: When multiple modalities are normalized (EEG, audio), ensure each modality uses its declared normalization method
  AVOID: No silently applying z-score to a min-max-specified modality

Validation

- Stats computation function receives only train indices.
- Validation data normalized with train-fitted stats (not re-fitted).
- Normalization method matches `spec.yaml` declaration for each modality.

Project Context (nn framework)

**Normalization sites in Exp04:**
- **z-score per window:** `nn::transforms::WindowZScore` / `EEGWindowZScore` (`include/utility/`), applied by the dataset loaders (`FsddWindowDataset.cpp`, `Meeting01MitBih.cpp`, ...) — each window is normalized with its OWN mean and std, so there is no fold statistic to leak ✓
- **per-window min-max for spike codes:** `encode_sample` in `src/experiments/meeting01/lib/src/Meeting01Encoding.cpp` — the window's own min and max, applied for `poisson` and `latency` only (`direct` presents the analog window unchanged); also per window, so no cross-sample statistic

**Leakage risk:** both sites above are per window and need no fit. Any NEW normalization that uses a statistic shared across windows (a dataset mean/std, a global min/max) must be fitted on the training fold only and then applied to validation; keep per-window where possible.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Key Files to Audit

- [include/nn/utility/Normalization.hpp](include/nn/utility/Normalization.hpp) — `normalize_0_1()`: check it accepts pre-computed stats
- [include/nn/utility/EEGWindowZScore.hpp](include/nn/utility/EEGWindowZScore.hpp) — z-score contract: fit separately per fold?
- [include/nn/utility/AudioMeanStdNormalize.hpp](include/nn/utility/AudioMeanStdNormalize.hpp) — mean-std contract

Correct Pattern

```cpp
// Fit on train indices only:
auto stats = compute_stats(dataset, train_indices);  // NOT all indices
NN_LOG_INFO("Norm stats: mean={}, std={}", stats.mean, stats.std);

// Apply same stats to both splits:
auto train_norm = apply_norm(dataset, train_indices, stats);
auto val_norm   = apply_norm(dataset, val_indices,   stats);  // same stats
```
