---
description: "Enforce deterministic experiment metadata, config serialization, and output traceability."
---

# experiment-reproducibility

Ensure every experiment run is reproducible and traceable.

## Rules

- **EXPLICIT_CONFIG**: Persist run config in JSON/YAML artifacts alongside results. No implicit defaults without serialization.
- **RESULT_LINKAGE**: Link each result to the exact profile/config used. No detached outputs.
- **TIMESTAMPED_OUTPUTS**: Timestamp output folders/files (ISO 8601: `YYYYMMDD_HHMMSS`). No overwriting prior results.
- **DETERMINISM**: Record seeds and deterministic settings (`torch.manual_seed`, OpenCL queue order, etc.). No non-repeatable runs.

## Required Artifact Structure

Each run must produce:
```
results/<experiment_id>/<YYYYMMDD_HHMMSS>/
├── config.json      ← full run configuration (hyperparams, paths, seeds)
├── metrics.json     ← quantitative results
└── <model>.bin      ← saved model state (optional but preferred)
```

## Validation

- Run artifacts include config + timestamp + result linkage metadata.
- Config file is human-readable JSON or YAML.
- No two runs share the same output directory.
