# Experiment00: Wavelet + Paraconsistent Baseline (Phase 0)

Frozen baseline pipeline establishing the reference accuracy for the thesis using
classical wavelet feature extraction and a paraconsistent classifier on the 10.1117
imagined speech dataset.

## Purpose

Provides the non-neural, interpretable baseline that all subsequent neural experiments
must beat. Uses Da Costa paraconsistent logic to assign credibility/plausibility pairs
to classification decisions — the novel contribution unique to this thesis.

## Dataset

10.1117 database: simultaneous EEG and audio recordings of imagined speech.
Loaded via `dataLoaders_10_1117`. Config selects subject, session, and trial windows.

## Pipeline

```
Raw EEG/audio trials
  → Wavelet packet decomposition (phase00_features.cpp)
  → Normalize features to [0, 1]
  → Paraconsistent metrics (Da Costa logic)
  → Train classifier
  → Write metrics / artifacts to results/
```

## Build and run

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target Phase00 -j$(nproc)

# Default config (hardcoded path)
./out/build/max-performance/src/experiments/00/Phase00

# Custom config
./out/build/max-performance/src/experiments/00/Phase00 path/to/config.json
```

## Key code paths

| File | Role |
|---|---|
| `src/experiments/00/phase00.cpp` | Entry point, orchestration |
| `src/experiments/00/phase00_data.cpp` | Trial extraction |
| `src/experiments/00/phase00_features.cpp` | Wavelet feature extraction |
| `src/experiments/00/phase00_training.cpp` | Classifier training and artifact writing |
| `include/paraconsistent/` | Da Costa paraconsistent logic |
| `include/wavelet/waveletOperations.h` | Wavelet packet decomposition |

## Important note

This experiment is intentionally frozen. Do not refactor or change default config
values without updating the thesis baseline numbers.

## See also

- [Paraconsistent Logic](../Concepts/Paraconsistent-Logic.md) — theory behind the classifier
- [Experiment02](Experiment02.md) — next step: replace paraconsistent classifier with autoencoder
- [Home](../Home.md)
