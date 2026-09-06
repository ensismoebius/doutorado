---
description: "Validate backward pass tensor shapes, cache consistency, and gradient finiteness before optimizer step."
---

# gradient-flow-validator

Catch silent backward pass failures: shape mismatches, stale caches, and NaN/Inf gradients that corrupt weight updates.

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

**Layer cache patterns** — check these are populated before `backward()`:
- `CrossEntropyLoss`: caches softmax output from `forward()`
- `LifBPTT`: `spike_history` length must equal `time_steps`; `v_mem_history` same
- `LinearImpl`: caches input tensor for weight gradient computation

**Time-major grad invariant:** Gradient entering `LifBPTT::backward()` must be `(T*B, F)` — same shape as forward input. If BPTT history length ≠ `time_steps`, the off-by-one means backward reads past the allocated history.

**Consequence of wrong history length:** Silent gradient corruption — spike gradients from wrong timestep are applied, loss appears to decrease but model diverges on longer sequences.

## Rules

- **SHAPE_MATCH**: Gradient input to `backward()` must have the same shape as the output of the corresponding `forward()`. Assert or log mismatch explicitly. No silent shape mismatches.
- **CACHE_VALIDITY**: Caches used in `backward()` (softmax output, spike history, activation masks) must be validated as non-empty and matching current batch dimensions. No use of stale cache from a previous forward call.
- **FINITE_BEFORE_STEP**: Check all gradient tensors for NaN/Inf after `backward()` and before `optimizer.step()`. Log which layer produced the first non-finite gradient. No updating weights with invalid gradients.
- **NORM_LOGGING**: When `clip_grad_norm` is applied, log the pre-clip gradient norm per layer in DEBUG mode. No silent clipping that hides exploding gradients.
- **BPTT_HISTORY_LENGTH**: For BPTT/SNN backward passes, assert that history length equals the number of unrolled time steps declared in the config. No off-by-one unrolling.
- **FORWARD_BEFORE_BACKWARD**: Assert that `forward()` was called before `backward()` on each layer. No calling `backward()` on an un-initialized cache.

## Key Files to Audit

- [include/layers/losses/CrossEntropyLoss.hpp](include/layers/losses/CrossEntropyLoss.hpp) — `last_targets` cache vs batch size
- [include/layers/spiking/LifBPTT.hpp](include/layers/spiking/LifBPTT.hpp) — `v_post_history`, `v_mem_history` consistency
- [include/layers/activations/LeakyReLU.hpp](include/layers/activations/LeakyReLU.hpp) — gradient mask shape
- [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — `clip_grad_norm` call site

## Checklist

1. For each layer with a `backward()`, confirm input shape == stored forward output shape.
2. Confirm all caches are cleared after optimizer step (prevent stale use across batches).
3. Add `NN_LOG_DEBUG` gradient norm output at the Trainer level.
4. Run a single training step and verify no NaN appears in any gradient tensor.

## Validation

- No shape mismatch between `backward()` input and `forward()` output in any layer.
- Gradient NaN/Inf triggers a log error and skips the optimizer step (or aborts training).
- BPTT history length assertion fires on misconfigured `time_steps`.
