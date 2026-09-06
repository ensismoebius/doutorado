---
name: initialization-determinism-enforcer
description: "Enforce seed propagation from experiment config through trainer into all layer parameter initializers."
---

# initialization-determinism-enforcer

Ensure every layer parameter initialization respects the `random_seed` declared in experiment config, making runs reproducible from a fixed seed.

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

**Initializer location:** `include/initializers/` — Glorot/Xavier, He, uniform, etc.

**Seed flow:**
`TrainerConfig::sampler_shuffle_seed` → `std::mt19937` in `Trainer` constructor → passed to data sampler and initializers

**Layer init timing:** Layer parameters are initialized in the constructor, not in `forward()`. Do not re-initialize in `forward()` — it resets learned weights every step.

**`seed_deterministic` field** in Exp04 profiles:
- `false` — article runs (random init, multiple runs averaged)
- `true` — debugging only (reproducible single run for diff comparison)

## Rules

- **SEED_MANDATORY**: Every experiment config must declare `random_seed`. No initialization without an explicit seed in the config.
- **SEED_PROPAGATED**: The seed must flow: `Config.random_seed` → `Trainer` constructor → initializer call site. No initializer called with `std::random_device{}` in production code paths.
- **NO_IMPLICIT_RNG**: Never call `std::random_device{}` in initializers during training. Reserve it only for optional one-time seed generation at CLI level (then record the generated seed).
- **LOG_SEED_AT_START**: Log the seed value at `INFO` level before any layer initialization. No unrecorded randomness.
- **RNG_STATE_SNAPSHOT**: For full reproducibility, log (or save) the full RNG state after initialization so it can be restored for debugging.
- **SINGLE_RNG**: Use a single `std::mt19937` instance seeded once from config, passed to all initializers. No multiple independent RNG instances that diverge.

## Key Files to Fix

- [include/initializers/xavier.hpp](include/initializers/xavier.hpp) — two branches: with/without seed
- [include/initializers/kaiming_snn.hpp](include/initializers/kaiming_snn.hpp) — `std::random_device` default
- [src/experiments/waveletAE/WaveletAEConfig.hpp](src/experiments/waveletAE/WaveletAEConfig.hpp) — `random_seed` declared but unused in init
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
