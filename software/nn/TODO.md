# Speaker Identification Pipeline: EEG + Audio with Spiking Neural Networks

## Daily TODO — Ultra-Concise Version

**Speaker Identification (EEG + Audio)**

---

## 🔒 PHASE 0 — Freezing (DO ONCE)
* [ ] Fix window **1.5 s / 50%**
* [ ] Fix normalization **[0,1] before α/β**
* [ ] Fix classifier (**ResNet**)
* [ ] Create unique `config.yaml`

---

## 🧱 PHASE 1 — Baseline Wavelet
* [ ] Pipeline **Wavelet / Wavelet-Packet** (experiments)
* [ ] Validate numerical examples of wavelet
* [ ] Calculate **α, β, G1, G2**
* [ ] Classify (ResNet)
* [ ] Save results (CSV)

---

## 🔬 PHASE 2 — LFCC × MEL × BARK (CENTRAL)
* [ ] Execute **LFCC × MEL × BARK**
* [ ] Modalities:
  * [ ] Voice
  * [ ] EEG
  * [ ] Voice + EEG
* [ ] Metrics:
  * [ ] α, β, G1, G2
  * [ ] Accuracy
* [ ] Consolidate comparative table

---

## 🧠 PHASE 3 — Autoencoders
* [ ] Sub-complete AE (experiments)
* [ ] Supra-complete AE (experiments)
* [ ] **Denoising AE** (mandatory)
* [ ] Fix architecture and bottleneck
* [ ] Compare:
  * [ ] AE × Wavelet-Packet
  * [ ] Classical × Regularized × Denoising
* [ ] Evaluate (α/β + accuracy)

---

## 🔀 PHASE 4 — Modalities (CLOSURE)
* [ ] Pronounced speech
* [ ] Imagined speech
* [ ] Mixed speech
* [ ] Compare:
  * [ ] Voice only
  * [ ] EEG only
  * [ ] Voice + EEG
* [ ] Demonstrate gain with:
  * [ ] Imagined speech
  * [ ] EEG + voice fusion

---

## 🛡️ PHASE 5 — Robustness (OPTIONAL)
* [ ] Insert controlled noise
* [ ] Measure drop in **G1** and accuracy
* [ ] Validate SNN tolerance

---

## 📊 PHASE 6 — Final Consolidation
* [ ] Table **LFCC × MEL × BARK**
* [ ] Table **Wavelet × AE × Denoising**
* [ ] Graph in paraconsistent plane
* [ ] Check traceability with Section 1.1.1

## ✅ **Implemented Components (/src/core/ and /src/demos/)**

### Core Neural Network Components
* [x] Implement complete **LIF** neuron.
* [x] Implement **surrogate gradient function** (Exponential / SuperSpike).
* [x] Implement **L1 and L2 regularization** (Appendix A).

### Signal Processing and Feature Extraction
* [x] Implement **DWT** via Mallat algorithm.
* [x] Implement **Wavelet-Packet Transform (WPT)**.
* [x] Select wavelets based on frequency and phase response.
* [x] Implement **BARK** (energies per band).
* [x] Implement **MEL / MFCC** (with DCT).
* [x] **Implement LFCC** (voice and EEG) - both signals implemented.
* [x] Calculate delta and delta-delta coefficients.

### Paraconsistent Analysis
* [x] Implement calculation of **α (intraclass similarity)**.
* [x] Implement calculation of **β (interclass overlap)**.
* [x] Calculate **G1** and **G2**.
* [x] Calculate distances to the optimal point **(1,0)** in the paraconsistent plane.
* [x] Use these values as **primary quality criterion for the vectors**.

### Autoencoders
* [x] Implement **sub-complete autoencoder** (in /src/demos/).
* [x] Implement **supra-complete autoencoder** (in /src/demos/).

---

## 📋 **Remaining Experimental Tasks (Not in /src/experiments/)**

### Experiment 01 - LFCC Pipeline Implementation
* [x] **LFCC / MEL / BARK Pipeline** - Implemented in `src/experiments/01/`
  * Complete LFCC extraction with preprocessing, framing, FFT, filterbank, DCT
  * Calculation of delta and delta-delta coefficients
  * Audio loading from .mat files
  * Subject processing orchestration

### Experimental Implementation Required
* [ ] Implement **denoising autoencoder** (Chap. 2.1.10.4) in /src/experiments/.
* [ ] Wavelet / Wavelet-Packet Pipeline in /src/experiments/.
* [ ] Mandatory normalization to **[0,1]** (paraconsistent prerequisite) in /src/experiments/.
* [ ] Classification with **Residual Neural Networks** in /src/experiments/.
* [ ] Cross-validation between:
  * different types of features;
  * different modalities (voice × EEG) in /src/experiments/.
* [ ] Report:
  * classical metrics (accuracy, etc.);
  * paraconsistent metrics in /src/experiments/.
* [ ] Demonstrate gains from:
  * imagined speech;
  * EEG + voice fusion in /src/experiments/.
* [ ] Analyze specific impact of **LFCC vs perceptual scales** in /src/experiments/.

---

## 🔍 **Validation and Comparison Tasks**

### Required Experimental Comparisons
* [ ] **Systematically compare LFCC × MEL × BARK** - mandatory experimental step
  * Comparison in terms of paraconsistent separability (α, β, G1, G2)
  * Classification performance
  * Robustness to severe speech degradations
* [ ] Compare:
  * wavelet-packet features × autoencoder features;
  * classical autoencoder × regularized × denoising.
* [ ] Validate SNN noise tolerance as discussed in Chap. 2.1.9.
* [ ] Ensure that the wavelet implementation reproduces the presented numerical examples.
* [ ] Ensure that all experiments are linked to the objectives listed in Section 1.1.1.

---

## Executive Summary

For the task of speaker identification using synchronized EEG and audio data, the recommended starting point is a **1.5-second window with a 50% overlap**. This duration is long enough to capture prosodic and intonational cues fundamental to speaker identity, which are often more discriminative than short-term phonetic features (Snyder et al., 2018). The 50% overlap ensures a good trade-off between temporal resolution and computational efficiency, generating a sufficient number of samples for training deep learning models without excessive redundancy.