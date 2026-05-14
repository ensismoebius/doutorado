# Experiment05: Biometric Authentication of Severely Dysphonic Speakers via Imagined Speech

> **Thesis primary experiment.**  
> Thesis: *Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech*  
> Author: André Furlan — UNESP | Advisor: Prof. Dr. Rodrigo Capobianco Guido | Funding: FAPESP 2021/12407-4

---

## Overview

Experiment05 implements the full speaker-authentication pipeline for individuals with severe laryngeal dysphonia (DLS). The core hypothesis: combining a degraded phonated voice with EEG-captured imagined speech produces a biometric signal that is both robust (works even when voice quality degrades) and difficult to spoof (EEG is not externally recordable without consent).

The experiment consists of two stages:

- **E3 — Feature Extraction**: handcrafted (DTWPT + classical descriptors, guided by paraconsistent quality ranking) vs. learned (LSTM-AE / SNN-AE)
- **E4 — Authentication**: RNN or DSNN classifier, text-dependent and text-independent modes, nested 5-fold cross-validation

---

## Theoretical Background

### Problem: Voice-Based Authentication Under Dysphonia

Conventional speaker-verification systems assume a clean, periodic voice signal. In severe laryngeal dysphonia, the phonation mechanism is impaired: aperiodic vibration, breathiness, and noise dominate the signal. Standard MFCC+GMM or x-vector systems degrade sharply because the formant structure that encodes speaker identity is obscured.

Imagined (covert) speech circumvents this: the speaker mentally rehearses an utterance without producing motor output. The same perisylvian network activates (Broca + Wernicke areas), and EEG captures the neural correlates. This neural signal is independent of laryngeal function — a severely dysphonic speaker produces identical imagined-speech EEG to a healthy speaker imagining the same utterance.

### Paraconsistent Feature Selection (EPC/α/β)

Before training any classifier, the pipeline ranks all feature-strategy×modality combinations using the paraconsistent quality metric. This avoids expensive hyperparameter sweeps.

Given N-class feature vectors normalised to [0,1]:

$$\alpha = 1 - \max_c \left( \frac{\overline{\max} - \overline{\min}}{\text{global range}} \right) \quad \text{(intraclass similarity)}$$

$$\beta = \frac{1}{N(N-1)} \sum_{i \neq j} R_{ij} / F \quad \text{(interclass overlap)}$$

Map to paraconsistent plane:

$$G_1 = \alpha - \beta \qquad G_2 = \alpha + \beta - 1$$

$$D_{\text{truth}} = \sqrt{(G_1 - 1)^2 + G_2^2}$$

Smaller $D_{\text{truth}}$ → features closer to the "Truth" corner → better speaker separability before any classifier sees them. The combination with the minimum $D_{\text{truth}}$ is passed to Stage E4.

See [Paraconsistent Feature Engineering](../Core/Paraconsistent.md) for the full derivation and API.

### Handcrafted Feature Extraction (DTWPT-based)

The Discrete-Time Wavelet Packet Transform decomposes the signal into a full binary tree of sub-bands. For a signal of length N with a J-level DWPT:

$$x[n] = \sum_{j,k} c_{j,k} \psi_{j,k}[n]$$

At each leaf node (sub-band), the following descriptors are computed:

| Descriptor | Formula | Sensitivity |
|---|---|---|
| Sub-band energy | $E_k = \sum_n c_k[n]^2$ | Overall power per band |
| ZCR | $\frac{1}{N}\sum_n \mathbb{1}[\text{sign}(x[n]) \neq \text{sign}(x[n-1])]$ | Voicing / noisiness |
| Entropy | $-\sum_k p_k \log p_k$ | Spectral spread |
| Teager–Kaiser | $\Psi[x(n)] = x^2(n) - x(n+1)x(n-1)$ | Amplitude modulation |
| Jitter | $\frac{\overline{|T_i - T_{i+1}|}}{\bar{T}}$ | Pitch period irregularity |
| Shimmer | $\frac{\overline{|A_i - A_{i+1}|}}{\bar{A}}$ | Amplitude irregularity |

Frequency scales evaluated: **BARK**, **MEL**, **LFCC** (see [LFCC](../Concepts/LFCC.md)).

### Learned Feature Extraction (Autoencoders)

**LSTM-AE**: sequence-to-sequence autoencoder. Encoder LSTM processes windowed frames, final hidden state = latent vector. Decoder LSTM reconstructs the frame sequence. Trained with MSE reconstruction loss + BPTT. See [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md).

**SNN-AE**: deep spiking autoencoder. `LifBPTTImpl` layers in encoder and decoder. Latent vector = mean spike rate over time window. Surrogate gradient (exponential) enables end-to-end training. Lower energy footprint than LSTM-AE. See [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md).

Both autoencoders are trained in an unsupervised manner — no speaker labels used. The learned latent representations then feed the paraconsistent ranking and subsequently the classifier.

### Authentication: Residual Network (RNN) and Deep SNN (DSNN)

**RNN** (here: Residual Neural Network, not recurrent):  
Skip connections prevent vanishing gradients in deep classifiers. Residual block:

$$\text{out} = F(x) + x$$

where $F$ = 2 Linear layers with BatchNorm + ReLU. Output layer: `linear:N_speakers:identity` → cross-entropy loss.

**DSNN**: deep spiking network classifier. Multiple `LifBPTT` + `ThresholdDependentBatchNorm` stages. Output: spike-count per class, decoded by `SpikeCountLoss`. Naturally sparse and energy-efficient at inference. See [Layers](../Core/Layers.md).

### Text-Dependent vs Text-Independent Evaluation

| Mode | Train phrases | Test phrases | Difficulty |
|---|---|---|---|
| Text-dependent | fixed set | same fixed set | Easier; high overlap |
| Text-independent | arbitrary | different arbitrary | Harder; tests generalisation |

Both modes use the same architecture; only the data split changes. Text-independent is the primary scientific contribution mode.

### Nested 5-Fold Cross-Validation

To avoid optimistic bias from hyperparameter tuning on the test fold, a nested k-fold scheme is used:

- **Outer loop** (5 folds): hold out one fold as test set; report final metrics here
- **Inner loop** (5 folds within training set): tune hyperparameters / early stopping

See [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md).

---

## Dataset

**10.1117/12.2255697** — EEG imagined speech (public):
- 15 Spanish-speaking subjects
- Utterances: vowels (`/a/ /e/ /i/ /o/ /u/`) + directional commands (`arriba/abajo/izquierda/derecha/adelante`)
- Three modalities: phonated speech, imagined speech, mixed
- Audio: 22050 Hz, 16-bit PCM WAV
- EEG: 800 Hz, 14 channels, 10-20 system (emotiv epoc)
- EEG preprocessing: bandpass 1–800 Hz, notch 60 Hz

See [Data Loaders](../Core/DataLoaders.md) for the `E05Dataset` loader API and file layout.

---

## Implementation

### Module plan

```
src/experiments/05/
├── experiment05.cpp              CLI entry point
├── lib/
│   ├── include/
│   │   ├── E05Config.hpp         profile JSON parser
│   │   ├── E05Dataset.hpp        EEG + voice loader (wraps 10.1117 loaders)
│   │   ├── E05FeatureExtraction.hpp  handcrafted + autoencoder pipeline
│   │   ├── E05Paraconsistent.hpp paraconsistent ranking step
│   │   ├── E05Classifiers.hpp    RNN / DSNN authentication
│   │   └── E05Output.hpp         CSV, JSON, DAT writers
│   └── src/
│       ├── E05Config.cpp
│       ├── E05Dataset.cpp
│       ├── E05FeatureExtraction.cpp
│       ├── E05Paraconsistent.cpp
│       ├── E05Classifiers.cpp
│       └── E05Output.cpp
├── profiles/
│   ├── debug.json
│   ├── handcrafted-eeg.json
│   ├── handcrafted-voice.json
│   ├── handcrafted-fused.json
│   ├── autoencoder-eeg.json
│   ├── autoencoder-voice.json
│   ├── autoencoder-fused.json
│   └── article-full.json
└── tests/
    ├── e05_profile_audit_gtest.cpp        48 tests — all 8 profiles parse+validate
    ├── e05_feature_extraction_gtest.cpp   descriptor functions + extract_handcrafted
    └── e05_classifiers_gtest.cpp          batch evaluate() + compute_aggregate_stats
```

### Profile schema

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
      "descriptors": ["energy", "zcr", "entropy", "teager", "jitter", "shimmer"]
    },
    "autoencoder": {
      "model": "lstm-ae",       // "lstm-ae" | "snn-ae"
      "encoder_layer_spec": ["linear:64:leaky", "linear:32:identity"],
      "decoder_layer_spec": ["linear:64:leaky", "linear:output:identity"]
    }
  },
  "paraconsistent": {
    "enabled": true
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

### Progress bars (three levels)

The experiment shows three concurrent progress bars during a run:

| Bar | Created by | Tracks |
|---|---|---|
| `Feature extraction` | `extract_features()` | samples processed (OpenMP parallel) |
| `Fold N/K \| <label>` | `ProgressCallback` (per outer fold) | epoch + batch within that fold |
| `E05 \| <run_tag>` | `experiment05.cpp` main | outer folds completed across all feature sets |

All bars are rendered by `nn::progress::ProgressManager` (background thread, ANSI escape codes). The global bar is completed and `ProgressManager::shutdown()` called before any output is written.

### Pipeline flow

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
            Rank all (strategy × modality × scale) combinations
            Select best — no classifier trained yet
                    │
                    ▼
            Classifier (RNN or DSNN)
            Nested 5-fold cross-validation
            Text-dependent + text-independent splits
                    │
                    ▼
            Results: accuracy, EER, D_truth per combination
```

### Performance notes

**Parallel feature extraction** — `extract_features()` uses `#pragma omp parallel for schedule(dynamic, 4)` over samples. The DTWPT computation (`wavelets::malat`) has no global state, so parallelism is safe. Pre-sized `fs.vectors.resize(n_samples)` avoids `push_back` races; progress counter uses `#pragma omp atomic capture`.

**Pre-built dataset tensors** — `run_classifier()` builds `(N, D)` input and `(N, C)` one-hot target tensors once via `mutable_data_ptr()` before the outer fold loop. Each fold slices row views via `Tensor::row(idx)` — cheap view, no copy.

**True batch training** — `Trainer::fit_loop_supervised` stacks a `(B, D)` batch each iteration, doing one GPU kernel per layer instead of B tiny `(1, D)` kernels. Both `LinearImpl` and `CrossEntropyLoss` support arbitrary batch dimension. See [Training](../Core/Training.md#true-batch-supervised-training).

**Batch evaluate()** — test-set evaluation stacks all test samples into one `(N_test, D)` tensor, one `model.forward()` call, then inline argmax + one-vs-rest confusion matrix.

---

## Build and run

```bash
# Configure (once)
cmake --preset=max-performance

# Build experiment binary
cmake --build out/build/max-performance --target experiment05 -j$(nproc)

# Single profile
./out/build/max-performance/src/experiments/05/experiment05 \
  --config src/experiments/05/profiles/handcrafted-eeg.json

# Full article run
./scripts/pipeline/run_experiment05.sh
```

---

## Outputs

| File | Contents |
|---|---|
| `results/e05_*_metrics.csv` | Per-fold accuracy, EER, D_truth |
| `results/e05_*_paraconsistent.csv` | α, β, G₁, G₂, D_truth per (strategy × modality × scale) |
| `results/e05_*_summary.json` | Config, seed, aggregated stats |
| `data/e05_*_comparison.dat` | pgfplots DAT for thesis figures |

---

## Key differences from Experiment04

| Aspect | Experiment04 | Experiment05 |
|---|---|---|
| Purpose | Congress paper (SNN vs LSTM reconstruction) | Thesis primary experiment |
| Dataset | FSDD (spoken digits, 8 kHz, audio only) | 10.1117/12.2255697 (EEG + voice, 22050 Hz) |
| Task | Autoencoder reconstruction (MSE) | Speaker authentication (accuracy, EER) |
| Signals | Audio only | Voice + EEG (bimodal) |
| Feature selection | Fixed architecture sweep | Paraconsistent α/β ranking before any classifier |
| Classifier | Autoencoder reconstruction loss | RNN / DSNN authentication |
| Text modes | N/A | Text-dependent + text-independent |
| Target population | General | Severely dysphonic speakers (DLS) |

---

## Common pitfalls

1. **Normalise before paraconsistent.** α and β are range-based; unnormalised features give meaningless D_truth values.

2. **EEG and voice window sizes differ.** EEG at 800 Hz needs different window parameters than voice at 22050 Hz. Do not reuse the same `window_size` across modalities.

3. **`jitter`/`shimmer` require voiced frames.** Unvoiced frames produce undefined period estimates. Filter by voicing flag before computing perturbation measures.

4. **SNN autoencoder encoder spec must use `linear` only.** `conv1d`/`residual` entries in `encoder_layer_spec` throw `std::invalid_argument` at startup.

5. **Text-independent split must not leak phrases.** Train and test splits must use disjoint phrase sets, not just disjoint utterances of the same phrase.

6. **Nested CV inner fold must not see test fold.** Hyperparameter selection (early stopping patience, lr) must use inner-loop val loss only.

7. **SQLite float32 blobs.** The 10.1117 database stores audio/EEG blobs as `float32`. `AudioLoader` and `EEGLoader` detect the encoding by comparing `sqlite3_column_bytes()` against both `n * sizeof(float)` and `n * sizeof(double)`. If this check fails, the loader throws `"unexpected audio blob size"`. Do not assume the DB format — always let the loader detect it.

8. **`nn::Tensor` default is a 0-dim scalar** (`size()=1, rows()=0, cols()=0`). Guard audio/EEG samples with `rows() > 0 && cols() > 0`, not `size() > 0`.

9. **`discoverSubjects` regex must have a capturing group.** Use `"^S(\\d+)$"` not `".*"`. Without the group `regex_groups_matches[1]` throws `std::out_of_range` at runtime.

---

## See also

- [Paraconsistent Feature Engineering](../Core/Paraconsistent.md) — α/β/D_truth formulas and API
- [Paraconsistent — Plain](../Core/Plain/Paraconsistent.md) — accessible explanation
- [LFCC](../Concepts/LFCC.md) — frequency scale used in handcrafted features
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — neuroscience background
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — SNN-AE / DSNN theory
- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md) — LSTM-AE theory
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — nested CV
- [Data Loaders](../Core/DataLoaders.md) — 10.1117 loader API
- [Research Context](../Research-Context.md) — thesis goals and full pipeline
- [Experiment04](./Experiment04.md) — prior congress paper experiment

---

## References

[A] R. C. Guido, "Paraconsistent feature engineering," *Knowledge-Based Systems*, 2018.

[B] S. Zhao et al., "EEG-based imagined speech recognition using deep learning," *IEEE Trans. Neural Syst. Rehabil. Eng.*, 2021.

[C] Dataset 10.1117/12.2255697: D. Benitez et al., "EEG signal database for imagined and real speech," *Proc. SPIE*, 2016.

[D] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate gradient learning in spiking neural networks," *IEEE Signal Process. Mag.*, vol. 36, no. 6, pp. 51–63, 2019.
