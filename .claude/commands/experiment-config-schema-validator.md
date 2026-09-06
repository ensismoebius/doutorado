
# experiment-config-schema-validator

Goal
- Catch configuration errors at load time rather than at training time. Every experiment config must be validated against a schema before any training begins.

Rules

- RULE: SCHEMA_REQUIRED
  DO: Each experiment must have a `schema.json` (JSON Schema draft-07) adjacent to its `spec.yaml`/`spec.json`
  AVOID: No config loading without schema validation
- RULE: REQUIRED_FIELDS
  DO: Validate all required fields are present: `random_seed`, `k_folds`, `classifier_paradigm`, `normalization_method`, `fit_on`
  AVOID: No silent defaults for missing required fields
- RULE: ENUM_VALIDATION
  DO: Validate categorical fields against allowed values: - `paradigm`: `["spiking_neural_network", "lstm", "autoencoder"]` - `normalization_method`: `["min_max", "z_score", "mean_std"]` - `fit_on`: `["train_only", "full_dataset"]`
- RULE: CROSS_FIELD_DEPS
  DO: Enforce field dependencies (e.g., if `paradigm == "spiking_neural_network"` then `surrogate_gradient` must be specified)
  AVOID: No invalid config combinations
- RULE: YAML_JSON_EQUIVALENCE
  DO: If both `spec.yaml` and `spec.json` exist, validate they produce equivalent parsed objects
  AVOID: No divergence between the two formats
- RULE: FAIL_FAST
  DO: Abort with a clear diagnostic on first validation failure
  AVOID: Never start training with an invalid config

Validation

- Invalid config (missing `random_seed`, bad `paradigm` value) aborts with a clear error before any layer is constructed.
- `spec.yaml` and `spec.json` parse to equivalent objects (diff their JSON representations).
- `schema.json` passes `jsonschema` meta-schema validation.

Project Context (nn framework)

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

**5 article profiles:** `src/experiments/guayaquil/profiles/article-{lstm-ae,snn-dense,snn-conv1d,snn-recurrent}.json` + `article-backend-bench.json`

**Profile parser:** `src/experiments/guayaquil/lib/include/ComparativeConfig.hpp`

**Code intelligence (MCP `code_intelligence`) — prefer over grep/manual commands for anything about the code itself:**
- `find_symbol` / `search_text` / `list_symbols` — resolve/search/enumerate symbols in indexed files, each hit tagged with its enclosing symbol (replaces `rg`/`grep`/`find` for anything already indexed)
- `get_source_range` / `symbol_source` / `outline_symbol` — exact, budget-checked source instead of a full-file read (`{"truncated": true, "recommended_ranges": [...]}` on overflow — read what it recommends, don't guess smaller)
- `find_references` / `find_dependencies` — callers/callees marked `"exact"` (real compiler) or `"heuristic"` (name-matching) — never read a heuristic "0 callers" as dead code
- `get_violations` / `rank_symbols` / `rename_symbol` — structural findings, complexity hotspots, and gated multi-site renames
- `ast_search` / `ast_replace` — AST-pattern structural search and rewrite (`foo($A, $B)` matches a 2-arg call to `foo` regardless of formatting/argument names) — prefer over a regex `search_text`/`rg` for anything shaped like code structure rather than text
- `run_build` / `run_tests` / `run_lint` / `run_format` / `detect_toolchain` — structured build/test/lint output, not raw logs (`run_lint`/`run_format` cover Python only; C++ still goes through `analysis-all`/`clang-format-changed.sh`)
- `git_status` / `git_log` / `git_blame` / `git_diff_stat` / `compare_baseline` — repo state/history/diff without shelling out to `git`

Validation Workflow

```bash
# Using Python jsonschema (available in pydemos env):
python -m jsonschema -i spec.json schema.json

# Or via jq for quick field checks:
jq '.random_seed, .classifier_paradigm, .normalization_method' spec.json
```

Key Files

- [src/experiments/waveletAE/spec.yaml](src/experiments/waveletAE/spec.yaml) — reference config to derive schema from
- [src/experiments/waveletAE/spec.json](src/experiments/waveletAE/spec.json) — JSON variant (check equivalence)
- [src/experiments/waveletAE/WaveletAEConfig.cpp](src/experiments/waveletAE/WaveletAEConfig.cpp) — add schema validation call at load time
- [src/experiments/guayaquil/profiles/](src/experiments/guayaquil/profiles/) — each profile needs schema validation
