# Experiment 05 — Thesis: Biometric Authentication of Severely Dysphonic Speakers via Imagined Speech

This is the **primary thesis experiment**. It implements the full pipeline for biometric speaker
authentication targeting individuals with severe laryngeal dysphonia (DLS), combining degraded
voice signals with EEG-captured imagined speech.

**Thesis:** Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech  
**Author:** André Furlan — UNESP  
**Advisor:** Prof. Dr. Rodrigo Capobianco Guido  
**Funding:** FAPESP 2021/12407-4

---

## Objective

Design and evaluate biometric algorithms that authenticate speakers who can only produce
potentially degraded speech, by fusing acoustic voice features with EEG imagined-speech features.

---

## What this experiment does

### Stage E3 — Feature Extraction (two strategies, compared)

**Handcrafted extraction** guided by Paraconsistent Feature Engineering (EPC/α/β):
- DTWPT (Discrete-Time Wavelet Packet Transform) sub-band energy
- ZCR (Zero-Crossing Rate)
- Entropy
- Teager–Kaiser energy operator
- Jitter, shimmer, perturbation measures
- Evaluated on BARK / MEL / LFCC frequency scales

**Feature learning** via autoencoders:
- LSTM-AE
- SNN-AE (Deep Spiking Neural Network autoencoder)

Both strategies applied to:
- Phonated speech signal (22050 Hz WAV, 16-bit)
- EEG imagined-speech signal (800 Hz, 14 channels, 10-20 system)

The paraconsistent α/β metric ranks all (feature strategy × signal modality) combinations by
D_truth before any classifier is trained — selecting the best feature set without expensive sweeps.

### Stage E4 — Authentication

Classifiers tested:
- **RNN** (Residual Neural Network)
- **DSNN** (Deep Spiking Neural Network)

Evaluation modes:
- **Text-dependent**: same phrase spoken and imagined at train and test time
- **Text-independent**: arbitrary utterances at train and test time

Primary dataset: `10.1117/12.2255697` (15 Spanish-speaking subjects, vowels + directional commands,
three modalities: phonated / imagined / mixed).

---

## Directory layout

```
05/
├── README.md               ← this file
├── CMakeLists.txt
├── experiment05.cpp        ← CLI entry point
├── lib/
│   ├── include/
│   │   ├── E05Config.hpp          ← profile JSON parser
│   │   ├── E05Dataset.hpp         ← EEG + voice loader
│   │   ├── E05FeatureExtraction.hpp  ← handcrafted + autoencoder pipeline
│   │   ├── E05Paraconsistent.hpp  ← paraconsistent selection step
│   │   ├── E05Classifiers.hpp     ← RNN / DSNN authentication
│   │   └── E05Output.hpp          ← CSV, JSON, DAT writers
│   └── src/
│       ├── E05Config.cpp
│       ├── E05Dataset.cpp
│       ├── E05FeatureExtraction.cpp
│       ├── E05Paraconsistent.cpp
│       ├── E05Classifiers.cpp
│       └── E05Output.cpp
├── profiles/
│   ├── debug.json                  ← fast smoke test (few epochs)
│   ├── handcrafted-eeg.json        ← DTWPT+paraconsistent on EEG only
│   ├── handcrafted-voice.json      ← DTWPT+paraconsistent on voice only
│   ├── handcrafted-fused.json      ← DTWPT+paraconsistent on voice+EEG
│   ├── autoencoder-eeg.json        ← LSTM-AE/SNN-AE on EEG only
│   ├── autoencoder-voice.json      ← LSTM-AE/SNN-AE on voice only
│   ├── autoencoder-fused.json      ← LSTM-AE/SNN-AE on voice+EEG
│   └── article-full.json           ← full comparison (all strategies × modalities)
└── tests/
    └── e05_profile_audit_gtest.cpp
```

---

## Build

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target experiment05 -j$(nproc)
```

## Run

```bash
# Single profile
./out/build/max-performance/src/experiments/05/experiment05 \
  --config src/experiments/05/profiles/handcrafted-eeg.json

# Full article run
./scripts/pipeline/run_experiment05.sh
```

---

## Profile schema (planned)

```jsonc
{
  "experiment": {
    "run_tag": "e05_handcrafted_eeg",
    "seed": 42,
    "repeats": 3,
    "seed_deterministic": false
  },
  "dataset": {
    "root": "/path/to/10.1117/",
    "results_dir": "results/",
    "modality": "eeg"          // "voice" | "eeg" | "fused"
  },
  "feature_extraction": {
    "strategy": "handcrafted",  // "handcrafted" | "autoencoder"
    "handcrafted": {
      "transform": "dtwpt",     // "dtwpt" | "lfcc" | "mfcc"
      "scale": "lfcc",          // "bark" | "mel" | "lfcc"
      "descriptors": ["energy", "zcr", "entropy", "teager"]
    },
    "autoencoder": {
      "model": "lstm-ae",       // "lstm-ae" | "snn-ae"
      "encoder_layer_spec": ["linear:64:leaky", "linear:32:identity"],
      "decoder_layer_spec": ["linear:64:leaky", "linear:output:identity"]
    }
  },
  "paraconsistent": {
    "enabled": true             // rank feature sets by D_truth before classification
  },
  "classifier": {
    "type": "rnn",              // "rnn" | "dsnn"
    "layer_spec": ["linear:128:relu", "residual:2", "linear:N_speakers:identity"],
    "text_mode": "dependent"    // "dependent" | "independent"
  },
  "training": {
    "epochs": 50,
    "learning_rate": 1e-3,
    "samples_per_batch": 32,
    "early_stop_patience": 10,
    "k_folds": 5,
    "nested_cv": true
  }
}
```

---

## Pipeline overview

```
10.1117/12.2255697 dataset
  │
  ├── Phonated speech (22050 Hz WAV)
  │     └── preprocessing (normalization, pre-emphasis, windowing)
  │           ├── Handcrafted: DTWPT + ZCR + entropy + Teager + jitter/shimmer
  │           └── Learned: LSTM-AE or SNN-AE → latent vectors
  │
  └── Imagined speech EEG (800 Hz, 14 ch)
        └── preprocessing (bandpass 1–800 Hz, notch 60 Hz)
              ├── Handcrafted: DTWPT energy per EEG band (alpha/beta/theta)
              └── Learned: LSTM-AE or SNN-AE → latent vectors
                    │
                    ▼
            Paraconsistent evaluation (α/β → D_truth)
            Select best (strategy × modality × scale) combination
                    │
                    ▼
            Classifier (RNN or DSNN)
            Nested 5-fold cross-validation
                    │
                    ▼
            Results: accuracy, EER, text-dep vs text-indep
```

---

## Outputs

| File | Contents |
|---|---|
| `results/e05_*_metrics.csv` | Per-fold accuracy, EER, D_truth |
| `results/e05_*_paraconsistent.csv` | α, β, G1, G2, D_truth per (strategy × modality × scale) |
| `results/e05_*_summary.json` | Config, seed, aggregated stats |
| `data/e05_*_comparison.dat` | pgfplots DAT for thesis figures |

---

## Key differences from Experiment 04

| Aspect | Experiment04 | Experiment05 |
|---|---|---|
| Purpose | Congress paper (FSDD reconstruction) | Thesis primary experiment |
| Dataset | FSDD (digits, 8 kHz) | 10.1117/12.2255697 (EEG + voice, 22050 Hz) |
| Task | Autoencoder reconstruction | Speaker authentication |
| Signals | Audio only | Voice + EEG (bimodal) |
| Feature selection | Fixed | Paraconsistent α/β ranking |
| Classifier | Autoencoder loss | RNN / DSNN authentication |
| Text modes | N/A | Text-dependent + text-independent |

---

## See also

- [Experiment05 wiki](./../../../.wiki/Experiments/Experiment05.md)
- [Research Context](./../../../.wiki/Research-Context.md)
- [Core/Paraconsistent](../../.wiki/Core/Paraconsistent.md)
- [Concepts/Imagined-Speech-and-EEG](../../.wiki/Concepts/Imagined-Speech-and-EEG.md)
- [Concepts/LFCC](../../.wiki/Concepts/LFCC.md)
