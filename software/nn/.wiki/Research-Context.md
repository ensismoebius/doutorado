# Research Context

This library is the software backbone of the doctoral thesis:

> **Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech**
> André Furlan — UNESP, advisor Prof. Dr. Rodrigo Capobianco Guido

---

## Problem Statement

Speaker verification systems rely on voice characteristics that are degraded or absent in individuals with **severe dysphonia** (hoarseness, aphonia, laryngectomy, etc.). Conventional MFCC-based systems fail on these speakers because the acoustic signal is unintelligible.

The thesis proposes augmenting degraded-voice biometrics with **EEG signals captured during imagined speech** — the cortical activity that precedes and accompanies speech production. Because imagined speech activates the full perisylvian language network regardless of the vocal tract condition, it provides a speaker-identifying signal even when overt phonation is impossible.

---

## General Objective

Design and implement biometric algorithms capable of authenticating individuals who can only produce potentially degraded speech, by adding EEG imagined-speech signals to the acoustic voice information.

### Current executable baseline (Experiment05)

- E3 handcrafted: DTWPT + descriptors, swept in Phase 00
- E3 automatic: SNN-AE and ANN-AE implemented and swept in Phase 00 (`autoencoder.model`)
- E3 automatic PCA (linear reference): **implemented 2026-11-11** (`feature_extraction.strategy = "pca"`, profiles `p00_pca_k{8,16,32}_{voice,eeg}`); not yet run
- LSTM-AE: still accepted by the code (`autoencoder.model = "lstm-ae"`), kept on purpose, used by no profile; it is no longer the default — since 2026-11-11 no thesis config key has a default (every key the run reads must be declared)
- E4 classifiers: RNN and DSNN implemented in pipeline

### Feature extractors of the thesis (decided 2026-11-11)

**PCA, ANN-AE, SNN-AE and DTWPT handcrafted** — linear, non-linear, spiking and hand-designed.

| Extractor | Role | Evidence behind the choice |
|---|---|---|
| PCA ($k$ = latent dim.) | A classic, well-established feature extractor (Jolliffe 2002): the $k$ directions of largest variance, in closed form, on the autoencoders' own input; the linear reference the learned extractors must beat. No training, so it separates "bad representation" from "bad training" in Phase 00's collapsed latents | Beat both autoencoders on reconstruction on all four meeting01 datasets |
| ANN-AE | Learned non-linear extractor | Beat the SNN-AE by 0.039–0.110 test MSE at 4.4–7.2× less energy (meeting01) |
| SNN-AE | Spiking extractor (feeds the DSNN path) | Search's mode: Poisson, conv1d input transform, $T$ ≈ 20–25, 2 layers |
| DTWPT handcrafted | Hand-designed baseline | Phase 00 winner on both signals by the paraconsistent criterion |

**Why no sequence autoencoder.** LSTM-, GRU- and Transformer-AE were pre-validated on
meeting01's development folds (FSDD, EEGMMIDB) and dropped. Quality first: none beat PCA, and the
LSTM-AE sat near the mean predictor (test MSE 0.967 on FSDD vs PCA 0.675; 0.821 ± 0.255 on
EEGMMIDB vs 0.192). Cost second: 257–338× the ANN-AE's energy per window, and 30–55× the
SNN-AE fold's wall time. The evidence is scoped — reduced search budget, development folds,
reconstruction, 256-sample windows — so the LSTM-AE stays in the code to be re-run if asked.
Thesis text: `chapters/08-proposedApproach.tex` ("Por que não autoencoders de sequência").

**Caveat that stays open:** reconstruction is not speaker identity. Which latent separates
speakers is decided in Phase 00 (paraconsistent α/β) and Phase 01 (EER), not by meeting01.

---

## Objectives

**1. Literature review**  
Survey state-of-the-art in biometric speaker authentication (ABL) with severe laryngeal dysphonias (DLS) + BCI imagined-speech decoding strategies.

**2. Study public databases**  
Use the public EEG imagined-speech dataset `10.1117/12.2255697` to enable initial experiments and comparison with prior work.

**3. Feature extraction — two approaches compared**  
For both voice and imagined-speech (EEG) signals:
- *Handcrafted extraction*: DTWPT sub-band energy, ZCR, entropy, Teager operator, jitter, shimmer, perturbation measures — guided by **paraconsistent feature engineering (EPC/α/β)**, which evaluates feature quality before any classifier is trained
- *Feature learning*: PCA (linear reference), ANN-AE and SNN-AE — investigate architecture topology and latent dimension (LSTM-AE pre-validated and dropped; see above)

**4. Authentication classifiers**  
Authenticate speakers using feature vectors via **RNNs (Residual Neural Networks)** and **DSNNs (Deep Spiking Neural Networks)**.  
Demonstrate that EEG imagined-speech features complement degraded voice, achieving accuracy above single-modality baselines. Compare accuracy and training cost. Two evaluation modes:
- **Text-dependent**: same phrase spoken and imagined at train and test time
- **Text-independent**: arbitrary utterances at train and test time

**5. C/C++ implementation**  
All algorithms in C/C++ for off-line operation, and real-time where feasible.

**6. Disseminate results**  
Conferences: ICASSP, Interspeech, MLSP.  
Journals: *IEEE Signal Processing Magazine*, *IEEE TPAMI*, *Neurocomputing*, *Computers in Biology and Medicine*.

---

## Pipeline

```
                    ┌────────────────────────────────────────────┐
                    │            Input Signals                   │
                    │  Speech (44100 Hz)     EEG (1024 Hz, 6ch) │
                    └──────────────┬──────────────┬─────────────┘
                                   │              │
                         ┌─────────▼──────┐  ┌───▼──────────────┐
                         │  DTWPT + LFCC  │  │  DTWPT + per-     │
                         │   filterbank   │  │  window z-score   │
                         └────────┬───────┘  └───────┬───────────┘
                                  │                  │
                         ┌────────▼──────────────────▼──────────┐
                         │      Paraconsistent Evaluation        │
                         │   α (intraclass) + β (interclass)     │
                         │   → D_truth ranks feature sets        │
                         └────────────────┬──────────────────────┘
                                          │  best (wavelet × scale)
                         ┌────────────────▼──────────────────────┐
                         │          Classifier                   │
                         │   DSNN / residual network (ResNet)     │
                         └────────────────┬──────────────────────┘
                                          │
                                   Authentication decision
```

### Feature extraction comparison

| Strategy | Signal | Method | Quality metric |
|---|---|---|---|
| Classical | Speech | DTWPT energy bands (BARK / MEL / LFCC) | Paraconsistent D_truth |
| Learned | Speech | PCA / ANN-AE / SNN-AE latent vectors | Paraconsistent D_truth |
| Classical | EEG | DTWPT energy bands (per band) | Paraconsistent D_truth |
| Learned | EEG | PCA / ANN-AE / SNN-AE latent vectors | Paraconsistent D_truth |
| Fusion | Speech + EEG | Concatenated or joint | Paraconsistent D_truth |

---

## Datasets

### 10.1117/12.2255697 (public validation)

- **15 Spanish-speaking subjects**
- Vowels (/a/, /e/, /i/, /o/, /u/) and directional commands
- Three modalities: phonated speech, imagined speech (EEG only), mixed
- Used for reproducibility and comparison with prior work
- Loader: `include/data_loaders/10.1117/`

---

## Novel Contributions

### Paraconsistent Feature Engineering

The primary methodological contribution. Given $N$ speaker classes and their feature vectors:

- **α** (intraclass similarity): measures how compact each class is — $1 - \max_n \overline{\text{svC}_n}$
- **β** (interclass overlap): fraction of cross-class component-value overlaps
- **D_truth** = $\sqrt{(G_1-1)^2 + G_2^2}$ — distance from the paraconsistent plane point $(\alpha-\beta,\; \alpha+\beta-1)$ to the "Truth" vertex $(1,0)$

Smaller D_truth → feature set naturally separable → good for any downstream classifier.

See [Core/Paraconsistent](./Core/Paraconsistent.md) for full theory and API.

### EEG Imagined Speech for Dysphonic Biometrics

Using EEG captured during imagined speech as a complement or substitute for degraded voice. The cortical activity pattern carries speaker identity even when the vocal tract cannot produce intelligible sound.

See [Concepts/Imagined-Speech-and-EEG](./Concepts/Imagined-Speech-and-EEG.md).

---

## Key Implementation Modules

| Module | Path | Role in thesis |
|---|---|---|
| Paraconsistent | `include/paraconsistent/` | Feature quality metric (novel contrib.) |
| Wavelet (DTWPT) | `include/wavelet/` | Classical feature extraction |
| Wave / LFCC | `include/wave/` | Linear filterbank for speaker verification |
| EEG loader | `include/data_loaders/10.1117/` | Public dataset I/O |
| SNN (LifBPTT) | `include/layers/spiking/` | Spiking neural network classifier |
| LSTM-AE / SNN-AE | `src/core/models/autoencoder/` | Learned feature extraction |
| Experiment 00 | `src/experiments/paraconsistentBaseline/` | DTWPT + paraconsistent baseline |
| Experiment 02 | `src/experiments/waveletAE/` | Wavelet autoencoder pipeline |
| Experiment 03 | `src/experiments/autoencoderRunner/` | Full audio/EEG autoencoder experiments |

---

## See Also

- [Core/Paraconsistent](./Core/Paraconsistent.md)
- [Core/Wavelet](./Core/Wavelet.md)
- [Core/Wave](./Core/Wave.md)
- [Core/DataLoaders](./Core/DataLoaders.md)
- [Concepts/LFCC](./Concepts/LFCC.md)
- [Concepts/Imagined-Speech-and-EEG](./Concepts/Imagined-Speech-and-EEG.md)
- [Concepts/SNN-and-Surrogate-Gradients](./Concepts/SNN-and-Surrogate-Gradients.md)
