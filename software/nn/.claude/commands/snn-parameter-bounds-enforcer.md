---
description: "Enforce physical bounds on SNN parameters (R, C, V_th) at the optimizer step, not silently at forward time."
---

# snn-parameter-bounds-enforcer

Prevent SNN parameter corruption: constrain R, C, and V_th before the optimizer writes them so that gradient computation is never based on clamped-but-wrong values.

## Project Context (nn framework)

**Current clamping location:** `include/nn/layers/spiking/LeakyBPTT.hpp` inside `forward()` — forward-time clamping as a safety net. Post-optimizer clamping (as this skill recommends) is not yet implemented.

**Current clamp values:**
- `R_min = C_min = 1e-6` — prevents division by zero in β = exp(−Δt/(R·C))
- `V_th`: convention ≥ 0.5 (enforced by construction/init, not dynamically clamped)

**β computation:** `β = exp(−Δt / (R·C))` — computed each forward step. If optimizer drives R or C negative → β > 1 → membrane diverges. Clamp prevents this.

**Grad zeroing:** Gradient w.r.t. R and C is zeroed when the forward clamp fires. Optimizer cannot pull them back from boundary — log a `WARN` if clamp fires frequently.

## Rules

- **CLAMP_BEFORE_STEP**: Apply parameter bounds as a post-step projection (after `optimizer.step()`, before `forward()`). Never rely solely on forward-time clamping to fix optimizer-proposed invalid values.
- **BOUNDS_EXPLICIT**: Document the valid range for each SNN parameter with physical justification:
  - `R` (resistance): `[1e-6, ∞)` — must be positive for RC circuit to be defined
  - `C` (capacitance): `[1e-6, ∞)` — must be positive
  - `V_th` (threshold): `(0, ∞)` — must be positive for meaningful threshold
- **BOUNDS_IN_CONFIG**: SNN parameter bounds must be declared in experiment config (not hardcoded per layer). Config: `snn.param_bounds: {R_min: 1e-6, C_min: 1e-6, V_th_min: 0.01}`.
- **LOG_VIOLATIONS**: When a parameter is projected back to its bound, log the violation at `WARN` level (layer name, parameter, proposed value, clamped value). No silent clamping.
- **GRADIENT_VALID_AFTER_CLAMP**: After projection, recompute or zero gradients for parameters that hit bounds. No stale gradients that point outside the feasible region.

## Key Files to Fix

- [include/nn/layers/spiking/LeakyBPTT.hpp](include/nn/layers/spiking/LeakyBPTT.hpp) — forward-time clamping of R/C (move to post-step projection)
- [include/nn/layers/spiking/Leaky.hpp](include/nn/layers/spiking/Leaky.hpp) — same pattern
- [include/nn/initializers/kaiming_snn.hpp](include/nn/initializers/kaiming_snn.hpp) — document valid initial ranges for V_th
- Optimizer step site in [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — add post-step projection call

## Post-Step Projection Pattern

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

## Validation

- SNN trains without forward-time clamping ever triggering (R, C, V_th stay in bounds after projection).
- Param-bound violations are logged with layer name and magnitude.
- Forward pass of SNN layers contains no clamping logic (projection handles it upstream).
