# Research Context

This library is the software backbone of the doctoral thesis:

> **Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech**
> André Furlan — UNESP, advisor Prof. Dr. Rodrigo Capobianco Guido

---

## Problem Statement

Speaker verification systems rely on voice characteristics that are degraded or absent in individuals with **severe dysphonia** (hoarseness, aphonia, laryngectomy, etc.). Conventional MFCC-based systems fail on these speakers because the acoustic signal is unintelligible.

The thesis proposes augmenting degraded-voice biometrics with **EEG signals captured during imagined speech** — the cortical activity that precedes and accompanies speech production. Because imagined speech activates the full perisylvian language network regardless of the vocal tract condition, it provides a speaker-identifying signal even when overt phonation is impossible.

---

## Thesis Goals

1. **Compare feature extraction strategies** for both speech and EEG:
   - DTWPT (Discrete-Time Wavelet Packet Transform) + paraconsistent feature quality evaluation
   - Autoencoder-based learned representations (LSTM-AE, SNN-AE)

2. **Evaluate feature quality** before classifier training using the novel paraconsistent α/β metric — avoids expensive grid sweeps by ranking feature sets geometrically.

3. **Classify and authenticate** speakers via deep residual/recurrent neural networks including spiking neural networks (SNN).

4. **Demonstrate that EEG imagined-speech features complement degraded voice** and together achieve authentication accuracy above single-modality baselines.

---

## Pipeline

```
                    ┌────────────────────────────────────────────┐
                    │            Input Signals                   │
                    │   Speech (8 kHz WAV)   EEG (800 Hz)       │
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

### Author's Own Dataset (thesis primary)

Collection protocol: two cycles per speaker (silent + noisy ambient)

Modalities:
1. Phonated speech — microphone, 8 kHz / 16-bit
2. Imagined speech — EEG only (no microphone), 800 Hz / 16-bit
3. Mixed — simultaneous phonation + EEG

Sentences (Portuguese):
- First name
- Directional commands: *cima / baixo / esquerda / direita*
- Invented password
- Common phrases: *"Estou com fome"*, *"Sinto dor"*, *"Entrar no sistema"*

Text modes:
- **Text-dependent**: same phrase spoken and imagined
- **Text-independent**: arbitrary utterances at train and test time

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
| SNN (LeakyBPTT) | `include/layers/spiking/` | Spiking neural network classifier |
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
