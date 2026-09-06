---
description: "Enforce consistent epsilon guards, NaN/Inf detection, and clamping across all forward/backward passes."
---

# numerical-stability-enforcer

Ensure all forward and backward passes use consistent, unified numerical safety patterns rather than ad-hoc per-layer strategies.

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

**Unified epsilon** — location: `include/tensor/Tensor.hpp` (or adjacent constants header), symbol `nn::kEps`. Use this everywhere; no inline `1e-6` literals.

**SNN clamp sites** — R and C are clamped to `≥1e-6` inside `LifBPTT::forward` (not post-optimizer). Grad is zeroed in the clamped region. If optimizer drives R or C negative, the clamp fires silently — add a `WARN` log if this happens frequently.

**Loss guards:**
- `SpikeCountLoss` / `SpikeTimeLoss`: `log(spike_count + kEps)` — guard against zero spikes
- `SpikeTimeLoss`: latency must be clamped to `[0, T-1]`; out-of-range = undefined behavior
- `CrossEntropyLoss`: `log(p + kEps)` in forward; already implemented

## Rules

- **UNIFIED_EPSILON**: Use a single project-wide epsilon constant (e.g., `nn::kEps = 1e-7f`) instead of per-file magic numbers. No inline `1e-6`, `1e-8`, or similar literals scattered across layers.
- **NAN_INF_GATE**: Check for NaN/Inf at the output of every forward pass in debug builds. Use `std::isfinite()` assertions or `NN_LOG_ERROR` + early return. No silent propagation of non-finite values.
- **CLAMP_EXPLICIT**: When clamping is needed (e.g., softmax, log, ReLU variants), document the mathematical reason and the bound. No unexplained `std::clamp(x, 1e-6f, ...)` without a comment on why that bound.
- **LOSS_GUARD**: Loss outputs must be finite. Detect and log non-finite loss before the optimizer step. No gradient updates when loss is NaN or Inf.
- **GRADIENT_FINITE**: After backward, check gradient tensors for NaN/Inf before applying optimizer. No propagating bad gradients into weight updates.
- **CLAMP_BEFORE_NOT_AFTER**: For physically bounded parameters (R, C, V_th in SNN), clamp proposed values before the optimizer writes them — not silently at forward time.

## Key Files to Audit

- [include/layers/losses/CrossEntropyLoss.hpp](include/layers/losses/CrossEntropyLoss.hpp) — numeric-stable softmax, epsilon guards
- [include/layers/losses/MSELoss.hpp](include/layers/losses/MSELoss.hpp) — NaN/Inf clipping
- [include/layers/spiking/LifBPTT.hpp](include/layers/spiking/LifBPTT.hpp) — R/C membrane clamping
- [include/layers/spiking/Lif.hpp](include/layers/spiking/Lif.hpp) — per-step parameter clamping
- [src/core/linear_algebra/linear_algebra.cpp](src/core/linear_algebra/linear_algebra.cpp) — raw numerical ops

## Audit Format

```
[FILE:LINE]
Issue: <what guard is missing or inconsistent>
Risk: (NaN propagation / precision loss / silent wrong result)
Fix: <unified pattern to apply>
```

## Validation

- No inline epsilon literals outside a shared constants header.
- Forward pass of each modified layer produces finite output on representative inputs.
- Loss value is checked for finiteness before every optimizer step.
