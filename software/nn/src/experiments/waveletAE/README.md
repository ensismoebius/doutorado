# Experiment 02 — Wavelet Autoencoder Pipeline

Config-driven pipeline that applies wavelet decomposition followed by an autoencoder
for EEG and audio signal reconstruction. Extends the Phase 0 baseline with a learned
encoder rather than a hand-crafted paraconsistent classifier.

## What it does

1. Loads EEG and audio signals from `.mat` files (paths in `spec.json`)
2. Applies wavelet packet decomposition
3. Trains a wavelet autoencoder (encoder + decoder layers)
4. Evaluates reconstruction quality and writes reporting artifacts

## Build

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target experiment_02 -j$(nproc)
```

## Run

```bash
# Default spec path (spec.json in src/experiments/02/)
./out/build/max-performance/src/experiments/02/experiment_02

# Custom spec
./out/build/max-performance/src/experiments/02/experiment_02 path/to/spec.json
```

## Config

`spec.json` / `spec.yaml` — experiment specification. Key fields:

- Dataset paths (EEG `.mat`, audio `.mat`)
- Wavelet parameters
- Autoencoder architecture
- Training hyperparameters (epochs, batch size, learning rate)

## Key source files

| File | Role |
|---|---|
| `experiment_02.cpp` | Thin entry point |
| `Experiment02Config.cpp` | Spec JSON/YAML loading |
| `Experiment02Data.cpp` | EEG/audio data loading |
| `Experiment02Wavelets.cpp` | Wavelet decomposition |
| `Experiment02Pipeline.cpp` | Top-level orchestration |
| `Experiment02Training.cpp` | Autoencoder training loop |
| `Experiment02Evaluation.cpp` | Reconstruction quality metrics |
| `Experiment02Reporting.cpp` | Result artifact writing |

## Tests

```bash
cmake --build out/build/max-performance --target experiment_02 -j$(nproc)
ctest --test-dir out/build/max-performance -R experiment_02 --output-on-failure
```
