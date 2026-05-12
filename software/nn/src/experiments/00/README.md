# Experiment 00 — Wavelet + Paraconsistent Baseline (Phase 0)

Frozen baseline pipeline. Establishes the reference accuracy for the thesis using
classical wavelet feature extraction followed by a paraconsistent classifier.

## What it does

1. Loads EEG/audio trials from the 10.1117 dataset via YAML config
2. Extracts wavelet packet features, normalizes to [0, 1]
3. Computes paraconsistent logic metrics (Da Costa framework)
4. Trains a small classifier and writes metrics/artifacts to `results/`

## Build

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target Phase00 -j$(nproc)
```

## Run

```bash
# Default config (hardcoded path, no CLI required)
./out/build/max-performance/src/experiments/00/Phase00

# Custom config
./out/build/max-performance/src/experiments/00/Phase00 path/to/config.json
```

## Key source files

| File | Role |
|---|---|
| `phase00.cpp` | Entry point — config load, top-level orchestration |
| `phase00_data.cpp` | Dataset traversal and trial extraction |
| `phase00_features.cpp` | Wavelet feature extraction |
| `phase00_training.cpp` | Classifier training loop and artifact writing |

## Dependencies

- `wavelet` — wavelet packet decomposition
- `paraconsistent` — Da Costa paraconsistent logic
- `data_loaders_10_1117` — dataset loader for 10.1117 EEG/audio
- `statistics` — k-fold, metrics

## Notes

No profiles directory — config is loaded from a YAML file (default path baked in).
This experiment is intentionally frozen; do not refactor without updating thesis baseline numbers.
