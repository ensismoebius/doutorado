---
description: "Enforce deterministic experiment metadata, config serialization, and output traceability."
---

# experiment-reproducibility

Ensure every experiment run is reproducible and traceable.

## Project Context (nn framework)

**Article pipeline chain** (full paper reproduction):
```
scripts/pipeline/run_article_profiles.sh
  → results/article_*_comparative_metrics.csv
  → scripts/pipeline/build_paper_data.py
  → documentation/.../data/article_*_*.dat
  → pdflatex paper.tex
```

**Profile locations:** `src/experiments/04/profiles/article-{lstm-ae,snn-dense,snn-conv1d,snn-recurrent}.json`

**Profile audit** — run after any profile edit:
```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

**`seed_deterministic` field** — set `false` in all article profiles (random init). Set `true` only for debugging/unit tests. Value recorded in CSV output for traceability.

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
