
# numerical-stability-enforcer

Goal
- Ensure all forward and backward passes use consistent, unified numerical safety patterns rather than ad-hoc per-layer strategies.

Rules

- RULE: UNIFIED_EPSILON
  DO: Use a single project-wide epsilon constant (e.g., `nn::kEps = 1e-7f`) instead of per-file magic numbers
  AVOID: No inline `1e-6`, `1e-8`, or similar literals scattered across layers
- RULE: NAN_INF_GATE
  DO: Check for NaN/Inf at the output of every forward pass in debug builds. Use `std::isfinite()` assertions or `NN_LOG_ERROR` + early return
  AVOID: No silent propagation of non-finite values
- RULE: CLAMP_EXPLICIT
  DO: When clamping is needed (e.g., softmax, log, ReLU variants), document the mathematical reason and the bound
  AVOID: No unexplained `std::clamp(x, 1e-6f, ...)` without a comment on why that bound
- RULE: LOSS_GUARD
  DO: Loss outputs must be finite. Detect and log non-finite loss before the optimizer step
  AVOID: No gradient updates when loss is NaN or Inf
- RULE: GRADIENT_FINITE
  DO: After backward, check gradient tensors for NaN/Inf before applying optimizer
  AVOID: No propagating bad gradients into weight updates
- RULE: CLAMP_BEFORE_NOT_AFTER
  DO: For physically bounded parameters (R, C, V_th in SNN), clamp proposed values before the optimizer writes them — not silently at forward time

Validation

- No inline epsilon literals outside a shared constants header.
- Forward pass of each modified layer produces finite output on representative inputs.
- Loss value is checked for finiteness before every optimizer step.

Project Context (nn framework)

**Unified epsilon** — location: `include/nn/tensor/Tensor.hpp` (or adjacent constants header), symbol `nn::kEps`. Use this everywhere; no inline `1e-6` literals.

**SNN clamp sites** — R and C are clamped to `≥1e-6` inside `LeakyBPTT::forward` (not post-optimizer). Grad is zeroed in the clamped region. If optimizer drives R or C negative, the clamp fires silently — add a `WARN` log if this happens frequently.

**Loss guards:**
- `SpikeCountLoss` / `SpikeTimeLoss`: `log(spike_count + kEps)` — guard against zero spikes
- `SpikeTimeLoss`: latency must be clamped to `[0, T-1]`; out-of-range = undefined behavior
- `CrossEntropyLoss`: `log(p + kEps)` in forward; already implemented

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Key Files to Audit

- [include/nn/layers/losses/CrossEntropyLoss.hpp](include/nn/layers/losses/CrossEntropyLoss.hpp) — numeric-stable softmax, epsilon guards
- [include/nn/layers/losses/MSELoss.hpp](include/nn/layers/losses/MSELoss.hpp) — NaN/Inf clipping
- [include/nn/layers/spiking/LeakyBPTT.hpp](include/nn/layers/spiking/LeakyBPTT.hpp) — R/C membrane clamping
- [include/nn/layers/spiking/Leaky.hpp](include/nn/layers/spiking/Leaky.hpp) — per-step parameter clamping
- [src/core/linearAlgebra/linear_algebra.cpp](src/core/linearAlgebra/linear_algebra.cpp) — raw numerical ops

Audit Format

```
[FILE:LINE]
Issue: <what guard is missing or inconsistent>
Risk: (NaN propagation / precision loss / silent wrong result)
Fix: <unified pattern to apply>
```
