
## MANDATORY FIRST STEP — Web search (automatic, no exceptions)

**Do this before anything else. Do NOT ask the user. Do NOT skip.**

1. Search official docs for every tool/API/component you will touch
2. Search for known bugs, changelogs, breaking changes
3. Search GitHub issues / forums for the exact error or behavior
4. Find working real-world examples

Training-data knowledge is outdated. Search first, implement second. Always.



# checkpoint-versioning-enforcer

Goal
- Ensure model checkpoints are self-describing: they carry the format version and architecture shape needed to detect and reject incompatible loads.

Rules

- RULE: VERSION_HEADER
  DO: Every checkpoint must begin with a format version string (e.g., `"nn_checkpoint_v1"`)
  AVOID: No loading a checkpoint without verifying this header first
- RULE: ARCHITECTURE_METADATA
  DO: Each checkpoint must include layer names, shapes (in, out), and dtypes
  AVOID: No loading weights into a model whose architecture was not validated against the saved metadata
- RULE: FAIL_ON_MISMATCH
  DO: When loading, if architecture metadata does not match the current model, fail loudly with a diagnostic
  AVOID: No silent loading of incompatible weights
- RULE: PROJECT_VERSION_TAG
  DO: Include the project CMake version (`0.2.0` or current) in the checkpoint metadata. Warn when checkpoint version predates current code
- RULE: ATOMIC_WRITE
  DO: Write checkpoints atomically (write to `<path>.tmp`, then rename)
  AVOID: No partial checkpoint files that can corrupt a training run
- RULE: COMPANION_JSON
  DO: Alongside the binary weight file, always write a `<name>.meta.json` with version, architecture, seed, and timestamp
  AVOID: No binary-only checkpoints

Validation

- Loading a checkpoint from a different architecture version prints an error and does not silently proceed.
- `model.meta.json` is present alongside every `model.bin`.
- Atomic write: no partial `.bin` file left on crash.

Project Context (nn framework)

**Checkpoint naming pattern:** `results/checkpoints/article_{model}_{backend}_{dataset}_{run_tag}.json`
Example: `results/checkpoints/article_lstm_ae_xtensor_fsdd_r01.json`

**Two serialization APIs:**
- `NetworkSerializer` (`include/nn/saver/NetworkSerializer.hpp`) — full `state_dict` map → `.npz` file; preferred for new code
- `NnSaver` (`include/nn/saver/NnSaver.hpp`) — legacy weight+bias pair → `_weights.npy` + `_bias.npy`; do not use for new layers

**Load pattern:**
```cpp
model.load_state_dict(NetworkSerializer::load("model.npz"));
```
Validate architecture metadata from the checkpoint against the current model before loading.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Expected Checkpoint Layout

```
checkpoints/<experiment_id>/<timestamp>/
├── model.bin           ← weight tensors
└── model.meta.json     ← { "format_version": "nn_checkpoint_v1",
                             "project_version": "0.2.0",
                             "timestamp": "...",
                             "random_seed": 42,
                             "layers": [{"name": "fc1", "shape": [128, 64], "dtype": "float32"}] }
```

Key Files to Update

- [include/nn/saver/NnSaver.hpp](include/nn/saver/NnSaver.hpp) — add version + metadata write
- [include/nn/saver/NetworkSerializer.hpp](include/nn/saver/NetworkSerializer.hpp) — add load-time compatibility check
