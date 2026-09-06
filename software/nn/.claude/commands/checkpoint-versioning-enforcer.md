---
description: "Enforce checkpoint format versioning, architecture metadata, and load-time compatibility validation."
---

# checkpoint-versioning-enforcer

Ensure model checkpoints are self-describing: they carry the format version and architecture shape needed to detect and reject incompatible loads.

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

**Checkpoint naming pattern:** `results/checkpoints/article_{model}_{backend}_{dataset}_{run_tag}.json`
Example: `results/checkpoints/article_lstm_ae_xtensor_fsdd_r01.json`

**Two serialization APIs:**
- `NetworkSerializer` (`include/serialization/NetworkSerializer.hpp`) — full `state_dict` map → `.npz` file; preferred for new code
- `NnSaver` (`include/serialization/NnSaver.hpp`) — legacy weight+bias pair → `_weights.npy` + `_bias.npy`; do not use for new layers

**Load pattern:**
```cpp
model.load_state_dict(NetworkSerializer::load("model.npz"));
```
Validate architecture metadata from the checkpoint against the current model before loading.

## Rules

- **VERSION_HEADER**: Every checkpoint must begin with a format version string (e.g., `"nn_checkpoint_v1"`). No loading a checkpoint without verifying this header first.
- **ARCHITECTURE_METADATA**: Each checkpoint must include layer names, shapes (in, out), and dtypes. No loading weights into a model whose architecture was not validated against the saved metadata.
- **FAIL_ON_MISMATCH**: When loading, if architecture metadata does not match the current model, fail loudly with a diagnostic. No silent loading of incompatible weights.
- **PROJECT_VERSION_TAG**: Include the project CMake version (`0.2.0` or current) in the checkpoint metadata. Warn when checkpoint version predates current code.
- **ATOMIC_WRITE**: Write checkpoints atomically (write to `<path>.tmp`, then rename). No partial checkpoint files that can corrupt a training run.
- **COMPANION_JSON**: Alongside the binary weight file, always write a `<name>.meta.json` with version, architecture, seed, and timestamp. No binary-only checkpoints.

## Expected Checkpoint Layout

```
checkpoints/<experiment_id>/<timestamp>/
├── model.bin           ← weight tensors
└── model.meta.json     ← { "format_version": "nn_checkpoint_v1",
                             "project_version": "0.2.0",
                             "timestamp": "...",
                             "random_seed": 42,
                             "layers": [{"name": "fc1", "shape": [128, 64], "dtype": "float32"}] }
```

## Key Files to Update

- [include/serialization/NnSaver.hpp](include/serialization/NnSaver.hpp) — add version + metadata write
- [include/serialization/NetworkSerializer.hpp](include/serialization/NetworkSerializer.hpp) — add load-time compatibility check

## Validation

- Loading a checkpoint from a different architecture version prints an error and does not silently proceed.
- `model.meta.json` is present alongside every `model.bin`.
- Atomic write: no partial `.bin` file left on crash.
