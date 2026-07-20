# WaveletAE: Wavelet Autoencoder Pipeline

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
  → Wavelet packet decomposition (WaveletAEWavelets.cpp)
  → Autoencoder training (WaveletAETraining.cpp)
  → Reconstruction quality evaluation (WaveletAEEvaluation.cpp)
  → Artifact writing (WaveletAEReporting.cpp)
```

## Config

`src/experiments/waveletAE/spec.json` (or `spec.yaml`). Key fields:

- `eeg_path`, `audio_path` — input `.mat` files
- Wavelet parameters (level, mother wavelet)
- Autoencoder architecture (layer widths)
- Training: epochs, batch size, learning rate

Pass a custom spec as the first CLI argument.

## Build and run

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target waveletAE -j$(nproc)

# Default spec path
./out/build/max-performance/src/experiments/waveletAE/waveletAE

# Custom spec
./out/build/max-performance/src/experiments/waveletAE/waveletAE path/to/spec.json
```

## Key code paths

| File | Role |
|---|---|
| `src/experiments/waveletAE/waveletAE.cpp` | Thin entry point |
| `src/experiments/waveletAE/WaveletAEConfig.cpp` | JSON/YAML spec loading |
| `src/experiments/waveletAE/WaveletAEData.cpp` | EEG/audio loading |
| `src/experiments/waveletAE/WaveletAEWavelets.cpp` | Wavelet decomposition |
| `src/experiments/waveletAE/WaveletAEPipeline.cpp` | Top-level orchestration |
| `src/experiments/waveletAE/WaveletAETraining.cpp` | Autoencoder training loop |
| `src/experiments/waveletAE/WaveletAEEvaluation.cpp` | Reconstruction metrics |
| `src/experiments/waveletAE/WaveletAEReporting.cpp` | Artifact writing |

## Tests

```bash
ctest --test-dir out/build/max-performance -R "WaveletAE.*" --output-on-failure
```

## See also

- [Experiment00](ParaconsistentBaseline.md) — paraconsistent baseline this replaces
- [AutoencoderRunner](AutoencoderRunner.md) — multimodal autoencoder with fused EEG+audio
- [Experiment04](Guayaquil.md) — LSTM vs SNN comparative study
- [Home](../Home.md)
