# WaveletAE: Wavelet Autoencoder Pipeline

Config-driven pipeline replacing the Phase 0 paraconsistent classifier with a learned
autoencoder. Applies wavelet decomposition to EEG and audio signals, then trains a
neural autoencoder for signal reconstruction.

## Purpose

Bridges the gap between the hand-crafted Phase 0 baseline and the full LSTM/SNN
comparative study (Experiment04). Validates that the wavelet preprocessing + neural
encoder approach is viable before introducing spiking dynamics.

## Dataset

EEG and audio `.mat` files of the Coretto base, named by `data.eeg_path` and `data.audio_path`
in the spec. There is no synthetic fallback: a missing file is an error.

## Pipeline

```
EEG + audio .mat files
  → Wavelet packet decomposition (WaveletAEWavelets.cpp)
  → Autoencoder training (WaveletAETraining.cpp)
  → Reconstruction quality evaluation (WaveletAEEvaluation.cpp)
  → Artifact writing (WaveletAEReporting.cpp)
```

## Config

The spec is a JSON file passed as the only argument; there is no default path and no YAML.
`src/experiments/waveletAE/spec.example.json` lists every key the loader reads
(`WaveletAEConfig.cpp`), all required:

| Key | Meaning |
|---|---|
| `experiment.id`, `experiment.determinism.random_seed` | run name; seed of the folds |
| `data.eeg_path`, `data.audio_path` | the EEG and audio `.mat` files (one Coretto subject, `SXX_EEG.mat` / `SXX_Audio.mat`) |
| `data.sampling_rate.EEG`, `data.sampling_rate.Audio` | 1024 and 44100 for the Coretto base |
| `data.segmentation.window.duration_sec`, `.overlap_sec` | window length and overlap |
| `paraconsistent_metrics.enabled` | compute α/β/G1/G2 |
| `validation_protocol.k` | folds of the cross-validation |

Every wavelet family of `get_available_wavelet_families()` is processed in turn. A file that
cannot be read, or a wavelet that yields no window, **stops the run** (until 2026-11-13 the
loader silently fell back to synthetic samples — with its placeholder paths it always did —
and skipped such wavelets).

## Build and run

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target waveletAE -j$(nproc)

cp src/experiments/waveletAE/spec.example.json my_spec.json   # then set data.eeg_path / audio_path
./out/build/max-performance/src/experiments/waveletAE/waveletAE my_spec.json
```

## Key code paths

| File | Role |
|---|---|
| `src/experiments/waveletAE/waveletAE.cpp` | Thin entry point |
| `src/experiments/waveletAE/WaveletAEConfig.cpp` | JSON spec loading |
| `src/experiments/waveletAE/WaveletAEData.cpp` | EEG/audio loading |
| `src/experiments/waveletAE/WaveletAEWavelets.cpp` | Wavelet decomposition |
| `src/experiments/waveletAE/WaveletAEPipeline.cpp` | Top-level orchestration |
| `src/experiments/waveletAE/WaveletAETraining.cpp` | Autoencoder training loop |
| `src/experiments/waveletAE/WaveletAEEvaluation.cpp` | Reconstruction metrics |
| `src/experiments/waveletAE/WaveletAEReporting.cpp` | Artifact writing |

## Tests

```bash
ctest --test-dir out/build/max-performance -R "WaveletAE" --output-on-failure   # suites WaveletAE{Data,Wavelets,Reporting}Test
```

## See also

- [Experiment00](ParaconsistentBaseline.md) — paraconsistent baseline this replaces
- [AutoencoderRunner](AutoencoderRunner.md) — multimodal autoencoder with fused EEG+audio
- [Meeting01](Meeting01.md) — the conference paper's SNN-AE comparison
- [Home](../Home.md)
