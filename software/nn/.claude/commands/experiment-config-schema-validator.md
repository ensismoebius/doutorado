---
description: "Validate experiment YAML/JSON configs against a JSON Schema: required fields, types, enums, and cross-field dependencies."
---

# experiment-config-schema-validator

Catch configuration errors at load time rather than at training time. Every experiment config must be validated against a schema before any training begins.

## Project Context (nn framework)

**Exp04 profile required fields** (validated by `profile_audit_gtest`):
- `experiment.run_tag` — unique identifier for CSV output
- `model.paradigm` — `"lstm"` or `"snn"`
- `training.batch_size`, `training.epochs`, `training.learning_rate`
- `loss` — must be `"mse"` for all article profiles
- `seed_deterministic` — must be `false` for article profiles

**Profile audit target:**
```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

**5 article profiles:** `src/experiments/04/profiles/article-{lstm-ae,snn-dense,snn-conv1d,snn-recurrent}.json` + `article-backend-bench.json`

**Profile parser:** `src/experiments/04/lib/include/ComparativeConfig.hpp`

## Rules

- **SCHEMA_REQUIRED**: Each experiment must have a `schema.json` (JSON Schema draft-07) adjacent to its `spec.yaml`/`spec.json`. No config loading without schema validation.
- **REQUIRED_FIELDS**: Validate all required fields are present: `random_seed`, `k_folds`, `classifier_paradigm`, `normalization_method`, `fit_on`. No silent defaults for missing required fields.
- **ENUM_VALIDATION**: Validate categorical fields against allowed values:
  - `paradigm`: `["spiking_neural_network", "lstm", "autoencoder"]`
  - `normalization_method`: `["min_max", "z_score", "mean_std"]`
  - `fit_on`: `["train_only", "full_dataset"]`
- **CROSS_FIELD_DEPS**: Enforce field dependencies (e.g., if `paradigm == "spiking_neural_network"` then `surrogate_gradient` must be specified). No invalid config combinations.
- **YAML_JSON_EQUIVALENCE**: If both `spec.yaml` and `spec.json` exist, validate they produce equivalent parsed objects. No divergence between the two formats.
- **FAIL_FAST**: Abort with a clear diagnostic on first validation failure. Never start training with an invalid config.

## Validation Workflow

```bash
# Using Python jsonschema (available in pydemos env):
python -m jsonschema -i spec.json schema.json

# Or via jq for quick field checks:
jq '.random_seed, .classifier_paradigm, .normalization_method' spec.json
```

## Key Files

- [src/experiments/02/spec.yaml](src/experiments/02/spec.yaml) — reference config to derive schema from
- [src/experiments/02/spec.json](src/experiments/02/spec.json) — JSON variant (check equivalence)
- [src/experiments/02/Experiment02Config.cpp](src/experiments/02/Experiment02Config.cpp) — add schema validation call at load time
- [src/experiments/04/profiles/](src/experiments/04/profiles/) — each profile needs schema validation

## Validation

- Invalid config (missing `random_seed`, bad `paradigm` value) aborts with a clear error before any layer is constructed.
- `spec.yaml` and `spec.json` parse to equivalent objects (diff their JSON representations).
- `schema.json` passes `jsonschema` meta-schema validation.
