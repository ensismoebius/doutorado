---
name: initialization-determinism-enforcer
description: "Enforce seed propagation from experiment config through trainer into all layer parameter initializers."
---

# initialization-determinism-enforcer

Ensure every layer parameter initialization respects the `random_seed` declared in experiment config, making runs reproducible from a fixed seed.

## Rules

- **SEED_MANDATORY**: Every experiment config must declare `random_seed`. No initialization without an explicit seed in the config.
- **SEED_PROPAGATED**: The seed must flow: `Config.random_seed` → `Trainer` constructor → initializer call site. No initializer called with `std::random_device{}` in production code paths.
- **NO_IMPLICIT_RNG**: Never call `std::random_device{}` in initializers during training. Reserve it only for optional one-time seed generation at CLI level (then record the generated seed).
- **LOG_SEED_AT_START**: Log the seed value at `INFO` level before any layer initialization. No unrecorded randomness.
- **RNG_STATE_SNAPSHOT**: For full reproducibility, log (or save) the full RNG state after initialization so it can be restored for debugging.
- **SINGLE_RNG**: Use a single `std::mt19937` instance seeded once from config, passed to all initializers. No multiple independent RNG instances that diverge.

## Key Files to Fix

- [include/nn/initializers/xavier.hpp](include/nn/initializers/xavier.hpp) — two branches: with/without seed
- [include/nn/initializers/kaiming_snn.hpp](include/nn/initializers/kaiming_snn.hpp) — `std::random_device` default
- [src/experiments/02/Experiment02Config.hpp](src/experiments/02/Experiment02Config.hpp) — `random_seed` declared but unused in init
- [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — seed must be threaded into layer init

## Propagation Pattern

```cpp
// In experiment entry point:
auto rng = std::mt19937{config.random_seed};
NN_LOG_INFO("RNG seed: {}", config.random_seed);

// Pass rng to every layer initializer:
xavier_init(layer.weights, rng);
kaiming_snn_init(layer.weights, rng);
```

## Validation

- Two runs with the same `random_seed` produce byte-identical initial weights.
- `std::random_device` does not appear in any initializer code path used during training.
- Seed value is logged before the first layer is constructed.
