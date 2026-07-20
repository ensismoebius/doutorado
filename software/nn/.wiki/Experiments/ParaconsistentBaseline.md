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
Loaded via `data_loaders_10_1117`. Config selects subject, session, and trial windows.

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
cmake --build out/build/max-performance --target paraconsistentBaseline -j$(nproc)

# Default config (hardcoded path)
./out/build/max-performance/src/experiments/paraconsistentBaseline/paraconsistentBaseline

# Custom config
./out/build/max-performance/src/experiments/paraconsistentBaseline/paraconsistentBaseline path/to/config.json
```

## Key code paths

| File | Role |
|---|---|
| `src/experiments/paraconsistentBaseline/phase00.cpp` | Entry point, orchestration |
| `src/experiments/paraconsistentBaseline/phase00_data.cpp` | Trial extraction |
| `src/experiments/paraconsistentBaseline/phase00_features.cpp` | Wavelet feature extraction |
| `src/experiments/paraconsistentBaseline/phase00_training.cpp` | Classifier training and artifact writing |
| `include/paraconsistent/` | Da Costa paraconsistent logic |
| `include/wavelet/waveletOperations.hpp` | Wavelet packet decomposition |

## Important note

This experiment is intentionally frozen. Do not refactor or change default config
values without updating the thesis baseline numbers.

## See also

- [Paraconsistent Logic](../Core/Paraconsistent.md) — theory behind the classifier
- [WaveletAE](WaveletAE.md) — next step: replace paraconsistent classifier with autoencoder
- [Home](../Home.md)
