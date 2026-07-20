
# initialization-determinism-enforcer

Goal
- Ensure every layer parameter initialization respects the `random_seed` declared in experiment config, making runs reproducible from a fixed seed.

Rules

- RULE: SEED_MANDATORY
  DO: Every experiment config must declare `random_seed`
  AVOID: No initialization without an explicit seed in the config
- RULE: SEED_PROPAGATED
  DO: The seed must flow: `Config.random_seed` → `Trainer` constructor → initializer call site
  AVOID: No initializer called with `std::random_device{}` in production code paths
- RULE: NO_IMPLICIT_RNG
  DO: Never call `std::random_device{}` in initializers during training. Reserve it only for optional one-time seed generation at CLI level (then record the generated seed)
- RULE: LOG_SEED_AT_START
  DO: Log the seed value at `INFO` level before any layer initialization
  AVOID: No unrecorded randomness
- RULE: RNG_STATE_SNAPSHOT
  DO: For full reproducibility, log (or save) the full RNG state after initialization so it can be restored for debugging
- RULE: SINGLE_RNG
  DO: Use a single `std::mt19937` instance seeded once from config, passed to all initializers
  AVOID: No multiple independent RNG instances that diverge

Validation

- Two runs with the same `random_seed` produce byte-identical initial weights.
- `std::random_device` does not appear in any initializer code path used during training.
- Seed value is logged before the first layer is constructed.

Project Context (nn framework)

**Initializer location:** `include/nn/initializers/` — Glorot/Xavier, He, uniform, etc.

**Seed flow:**
`TrainerConfig::sampler_shuffle_seed` → `std::mt19937` in `Trainer` constructor → passed to data sampler and initializers

**Layer init timing:** Layer parameters are initialized in the constructor, not in `forward()`. Do not re-initialize in `forward()` — it resets learned weights every step.

**`seed_deterministic` field** in Exp04 profiles:
- `false` — article runs (random init, multiple runs averaged)
- `true` — debugging only (reproducible single run for diff comparison)

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

Key Files to Fix

- [include/nn/initializers/xavier.hpp](include/nn/initializers/xavier.hpp) — two branches: with/without seed
- [include/nn/initializers/kaiming_snn.hpp](include/nn/initializers/kaiming_snn.hpp) — `std::random_device` default
- [src/experiments/waveletAE/WaveletAEConfig.hpp](src/experiments/waveletAE/WaveletAEConfig.hpp) — `random_seed` declared but unused in init
- [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — seed must be threaded into layer init

Propagation Pattern

```cpp
// In experiment entry point:
auto rng = std::mt19937{config.random_seed};
NN_LOG_INFO("RNG seed: {}", config.random_seed);

// Pass rng to every layer initializer:
xavier_init(layer.weights, rng);
kaiming_snn_init(layer.weights, rng);
```
