
# snn-parameter-bounds-enforcer

Goal
- Prevent SNN parameter corruption: constrain R, C, and V_th before the optimizer writes them so that gradient computation is never based on clamped-but-wrong values.

Rules

- RULE: CLAMP_BEFORE_STEP
  DO: Apply parameter bounds as a post-step projection (after `optimizer.step()`, before `forward()`)
  AVOID: Never rely solely on forward-time clamping to fix optimizer-proposed invalid values
- RULE: BOUNDS_EXPLICIT
  DO: Document the valid range for each SNN parameter with physical justification: - `R` (resistance): `[1e-6, ∞)` — must be positive for RC circuit to be defined - `C` (capacitance): `[1e-6, ∞)` — must be positive - `V_th` (threshold): `(0, ∞)` — must be positive for meaningful threshold
- RULE: BOUNDS_IN_CONFIG
  DO: SNN parameter bounds must be declared in experiment config (not hardcoded per layer). Config: `snn.param_bounds: {R_min: 1e-6, C_min: 1e-6, V_th_min: 0.01}`
- RULE: LOG_VIOLATIONS
  DO: When a parameter is projected back to its bound, log the violation at `WARN` level (layer name, parameter, proposed value, clamped value)
  AVOID: No silent clamping
- RULE: GRADIENT_VALID_AFTER_CLAMP
  DO: After projection, recompute or zero gradients for parameters that hit bounds
  AVOID: No stale gradients that point outside the feasible region

Validation

- SNN trains without forward-time clamping ever triggering (R, C, V_th stay in bounds after projection).
- Param-bound violations are logged with layer name and magnitude.
- Forward pass of SNN layers contains no clamping logic (projection handles it upstream).

Project Context (nn framework)

**Current clamping location:** `include/nn/layers/spiking/LeakyBPTT.hpp` inside `forward()` — forward-time clamping as a safety net. Post-optimizer clamping (as this skill recommends) is not yet implemented.

**Current clamp values:**
- `R_min = C_min = 1e-6` — prevents division by zero in β = exp(−Δt/(R·C))
- `V_th`: convention ≥ 0.5 (enforced by construction/init, not dynamically clamped)

**β computation:** `β = exp(−Δt / (R·C))` — computed each forward step. If optimizer drives R or C negative → β > 1 → membrane diverges. Clamp prevents this.

**Grad zeroing:** Gradient w.r.t. R and C is zeroed when the forward clamp fires. Optimizer cannot pull them back from boundary — log a `WARN` if clamp fires frequently.

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Key Files to Fix

- [include/nn/layers/spiking/LeakyBPTT.hpp](include/nn/layers/spiking/LeakyBPTT.hpp) — forward-time clamping of R/C (move to post-step projection)
- [include/nn/layers/spiking/Leaky.hpp](include/nn/layers/spiking/Leaky.hpp) — same pattern
- [include/nn/initializers/kaiming_snn.hpp](include/nn/initializers/kaiming_snn.hpp) — document valid initial ranges for V_th
- Optimizer step site in [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — add post-step projection call

Post-Step Projection Pattern

```cpp
// After optimizer.step():
for (auto& layer : snn_layers) {
    auto& R = layer.resistance;
    if (R < R_min) {
        NN_LOG_WARN("R clamped: {} → {}", R, R_min);
        R = R_min;
        layer.grad_R = 0.0f;  // zero gradient at boundary
    }
    // same for C, V_th
}
```
