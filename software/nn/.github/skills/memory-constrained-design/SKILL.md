---
name: memory-constrained-design
description: "Constrain model and pipeline choices to fit low-memory targets (Raspberry Pi class and similar)."
---

# memory-constrained-design

Keep experiments inside hard RAM budgets for embedded and edge targets.

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

**Target hardware:** AMD Renoir APU — 7 compute units, 64 KiB LDS per CU, 4 GB RAM shared between CPU and GPU. No discrete VRAM; all OpenCL buffers come from this pool.

**SNN memory cost:** `O(T × B × F)` per history buffer (`v_mem_history`, `spike_history`). For T=10, B=32, F=64: 20,480 floats = 80 KB per layer.

**Batch size choice:** `batch_size=32` chosen to keep all layer buffers + weights within GPU buffer pool. Larger batches (128+) engage more CUs but may exhaust the 4 GB shared pool.

**`set_gpu_resident(true)`** — keeps weight tensors in the GPU buffer pool between batches. Avoids repeated host→device copies. Set for all Linear layers in Exp04 by default.

## Rules

- **HARD_BUDGET**: Treat memory budget as a hard constraint, not a soft goal. Never optimize first and budget later.
- **SMALL_DEFAULTS**: Use `batch_size <= 8`, `latent <= 32`, `depth <= 2` unless explicitly justified otherwise.
- **DTYPE_DISCIPLINE**: Prefer `float32` for training/inference. No higher precision defaults on constrained devices.
- **TENSOR_CAP**: Keep single tensor allocations under practical limits (target: <10 MB). No silent large allocation spikes.

## Checklist

1. Check experiment config for budget-sensitive parameters (`batch_size`, `latent_dim`, `depth`, `time_steps`).
2. Estimate peak memory: sum of model weights + gradient buffers + activation tensors.
3. Flag any tensor allocation that could exceed 10 MB.
4. Confirm `dtype` is `float32` throughout.

## Validation

- Config declares budget-sensitive parameters explicitly.
- Memory-heavy operations are justified and measured.
- Peak memory estimate fits within declared hardware budget.
