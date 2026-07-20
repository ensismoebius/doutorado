# Experiment02: Wavelet Autoencoder Pipeline

Config-driven pipeline replacing the Phase 0 paraconsistent classifier with a learned
autoencoder. Applies wavelet decomposition to EEG and audio signals, then trains a
neural autoencoder for signal reconstruction.

## Purpose

Bridges the gap between the hand-crafted Phase 0 baseline and the full LSTM/SNN
comparative study (Experiment04). Validates that the wavelet preprocessing + neural
encoder approach is viable before introducing spiking dynamics.

## Dataset

EEG and audio `.mat` files; paths specified in `spec.json`. Synthetic fallback data
is generated automatically when real dataset paths are not configured.

## Pipeline

```
EEG + audio .mat files
  → Wavelet packet decomposition (Experiment02Wavelets.cpp)
  → Autoencoder training (Experiment02Training.cpp)
  → Reconstruction quality evaluation (Experiment02Evaluation.cpp)
  → Artifact writing (Experiment02Reporting.cpp)
```

## Config

`src/experiments/02/spec.json` (or `spec.yaml`). Key fields:

- `eeg_path`, `audio_path` — input `.mat` files
- Wavelet parameters (level, mother wavelet)
- Autoencoder architecture (layer widths)
- Training: epochs, batch size, learning rate

Pass a custom spec as the first CLI argument.

## Build and run

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target experiment_02 -j$(nproc)

# Default spec path
./out/build/max-performance/src/experiments/02/experiment_02

# Custom spec
./out/build/max-performance/src/experiments/02/experiment_02 path/to/spec.json
```

## Key code paths

| File | Role |
|---|---|
| `src/experiments/02/experiment_02.cpp` | Thin entry point |
| `src/experiments/02/Experiment02Config.cpp` | JSON/YAML spec loading |
| `src/experiments/02/Experiment02Data.cpp` | EEG/audio loading |
| `src/experiments/02/Experiment02Wavelets.cpp` | Wavelet decomposition |
| `src/experiments/02/Experiment02Pipeline.cpp` | Top-level orchestration |
| `src/experiments/02/Experiment02Training.cpp` | Autoencoder training loop |
| `src/experiments/02/Experiment02Evaluation.cpp` | Reconstruction metrics |
| `src/experiments/02/Experiment02Reporting.cpp` | Artifact writing |

## Tests

```bash
ctest --test-dir out/build/max-performance -R "Experiment02.*" --output-on-failure
```

## See also

- [Experiment00](Experiment00.md) — paraconsistent baseline this replaces
- [Experiment03](Experiment03.md) — multimodal autoencoder with fused EEG+audio
- [Experiment04](Experiment04.md) — LSTM vs SNN comparative study
- [Home](../Home.md)
