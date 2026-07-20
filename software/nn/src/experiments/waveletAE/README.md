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
cmake --build out/build/max-performance --target waveletAE -j$(nproc)
```

## Run

```bash
# Default spec path (spec.json in src/experiments/waveletAE/)
./out/build/max-performance/src/experiments/waveletAE/waveletAE

# Custom spec
./out/build/max-performance/src/experiments/waveletAE/waveletAE path/to/spec.json
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
| `waveletAE.cpp` | Thin entry point |
| `WaveletAEConfig.cpp` | Spec JSON/YAML loading |
| `WaveletAEData.cpp` | EEG/audio data loading |
| `WaveletAEWavelets.cpp` | Wavelet decomposition |
| `WaveletAEPipeline.cpp` | Top-level orchestration |
| `WaveletAETraining.cpp` | Autoencoder training loop |
| `WaveletAEEvaluation.cpp` | Reconstruction quality metrics |
| `WaveletAEReporting.cpp` | Result artifact writing |

## Tests

```bash
cmake --build out/build/max-performance --target waveletAE -j$(nproc)
ctest --test-dir out/build/max-performance -R waveletAE --output-on-failure
```
