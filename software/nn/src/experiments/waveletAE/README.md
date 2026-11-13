# Experiment 02 — Wavelet Autoencoder Pipeline

Config-driven pipeline that applies wavelet decomposition followed by an autoencoder
for EEG and audio signal reconstruction. Extends the Phase 0 baseline with a learned
encoder rather than a hand-crafted paraconsistent classifier.

## What it does

1. Loads EEG and audio signals from the `.mat` files the spec names (`data.eeg_path`, `data.audio_path`); a file that cannot be read stops the run
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
cp src/experiments/waveletAE/spec.example.json my_spec.json   # set the two data paths
./out/build/max-performance/src/experiments/waveletAE/waveletAE my_spec.json   # the spec is required
```

## Config

A JSON spec, the binary's only argument. Every key it reads is in
[`spec.example.json`](spec.example.json) and described in
[`.wiki/Experiments/WaveletAE.md` › Config](../../../.wiki/Experiments/WaveletAE.md#config).

## Key source files

| File | Role |
|---|---|
| `waveletAE.cpp` | Thin entry point |
| `WaveletAEConfig.cpp` | Spec JSON loading |
| `WaveletAEData.cpp` | EEG/audio data loading |
| `WaveletAEWavelets.cpp` | Wavelet decomposition |
| `WaveletAEPipeline.cpp` | Top-level orchestration |
| `WaveletAETraining.cpp` | Autoencoder training loop |
| `WaveletAEEvaluation.cpp` | Reconstruction quality metrics |
| `WaveletAEReporting.cpp` | Result artifact writing |

## Tests

```bash
cmake --build out/build/max-performance --target waveletAE -j$(nproc)
ctest --test-dir out/build/max-performance -R WaveletAE --output-on-failure   # suites are WaveletAE*Test
```
