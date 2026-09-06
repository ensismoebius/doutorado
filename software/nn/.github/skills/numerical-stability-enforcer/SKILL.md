---
name: numerical-stability-enforcer
description: "Enforce consistent epsilon guards, NaN/Inf detection, and clamping across all forward/backward passes."
---

# numerical-stability-enforcer

Ensure all forward and backward passes use consistent, unified numerical safety patterns rather than ad-hoc per-layer strategies.

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
- [include/layers/spiking/LeakyBPTT.hpp](include/layers/spiking/LeakyBPTT.hpp) — R/C membrane clamping
- [include/layers/spiking/Leaky.hpp](include/layers/spiking/Leaky.hpp) — per-step parameter clamping
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

Project Context (nn framework)
**Unified epsilon** — location: `include/tensor/Tensor.hpp` (or adjacent constants header), symbol `nn::kEps`. Use this everywhere; no inline `1e-6` literals.

**SNN clamp sites** — R and C are clamped to `≥1e-6` inside `LeakyBPTT::forward` (not post-optimizer). Grad is zeroed in the clamped region. If optimizer drives R or C negative, the clamp fires silently — add a `WARN` log if this happens frequently.

**Loss guards:**
- `SpikeCountLoss` / `SpikeTimeLoss`: `log(spike_count + kEps)` — guard against zero spikes
- `SpikeTimeLoss`: latency must be clamped to `[0, T-1]`; out-of-range = undefined behavior
- `CrossEntropyLoss`: `log(p + kEps)` in forward; already implemented
