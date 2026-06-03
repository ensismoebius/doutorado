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

- E3 automatic: LSTM-AE implemented in pipeline
- E3 automatic SNN-AE: planned (not yet wired in Experiment05 executable)
- E4 classifiers: RNN and DSNN implemented in pipeline

---

## Objectives

**1. Literature review**  
Survey state-of-the-art in biometric speaker authentication (ABL) with severe laryngeal dysphonias (DLS) + BCI imagined-speech decoding strategies.

**2. Study public databases**  
Use the public EEG imagined-speech dataset `10.1117/12.2255697` to enable initial experiments and comparison with prior work.

**3. Feature extraction — two approaches compared**  
For both voice and imagined-speech (EEG) signals:
- *Handcrafted extraction*: DTWPT sub-band energy, ZCR, entropy, Teager operator, jitter, shimmer, perturbation measures — guided by **paraconsistent feature engineering (EPC/α/β)**, which evaluates feature quality before any classifier is trained
- *Feature learning*: autoencoders (LSTM-AE, SNN-AE) — investigate architecture topology and latent dimension

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
                    │  Speech (22050 Hz WAV)  EEG (800 Hz)       │
                    └──────────────┬──────────────┬─────────────┘
                                   │              │
                         ┌─────────▼──────┐  ┌───▼──────────────┐
                         │  DTWPT + LFCC  │  │ Bandpass + Notch  │
                         │   filterbank   │  │  (1–800 Hz)       │
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
                         │   SNN residual / LSTM autoencoder      │
                         └────────────────┬──────────────────────┘
                                          │
                                   Authentication decision
```

### Feature extraction comparison

| Strategy | Signal | Method | Quality metric |
|---|---|---|---|
| Classical | Speech | DTWPT energy bands (BARK / MEL / LFCC) | Paraconsistent D_truth |
| Learned | Speech | LSTM-AE / SNN-AE latent vectors | Paraconsistent D_truth |
| Classical | EEG | DTWPT energy bands (per band) | Paraconsistent D_truth |
| Learned | EEG | LSTM-AE / SNN-AE latent vectors | Paraconsistent D_truth |
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
| Experiment 00 | `src/experiments/00/` | DTWPT + paraconsistent baseline |
| Experiment 02 | `src/experiments/02/` | Wavelet autoencoder pipeline |
| Experiment 03 | `src/experiments/03/` | Full audio/EEG autoencoder experiments |

---

## See Also

- [Core/Paraconsistent](./Core/Paraconsistent.md)
- [Core/Wavelet](./Core/Wavelet.md)
- [Core/Wave](./Core/Wave.md)
- [Core/DataLoaders](./Core/DataLoaders.md)
- [Concepts/LFCC](./Concepts/LFCC.md)
- [Concepts/Imagined-Speech-and-EEG](./Concepts/Imagined-Speech-and-EEG.md)
- [Concepts/SNN-and-Surrogate-Gradients](./Concepts/SNN-and-Surrogate-Gradients.md)
