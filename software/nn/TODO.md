# Speaker Identification Pipeline: EEG + Audio with Spiking Neural Networks

## Project Status and Tasks

### Daily TODO

**Speaker Identification (EEG + Audio)**

---

## 🔒 PHASE 0 — Freezing (DO ONCE)
* [ ] Fix window **1.5 s / 50%**
* [ ] Fix normalization **[0,1] before α/β**
* [ ] Mandatory normalization to **[0,1]** (paraconsistent prerequisite) in /src/experiments/.
* [ ] Fix classifier (**ResNet**)
* [ ] Classification with **Residual Neural Networks** in /src/experiments/.
* [ ] Create unique `config.yaml`

---

## 🧱 PHASE 1 — Baseline Wavelet
* [ ] Pipeline **Wavelet / Wavelet-Packet** (experiments)
* [ ] Wavelet / Wavelet-Packet Pipeline in /src/experiments/.
* [ ] Validate numerical examples of wavelet
* [ ] Ensure that the wavelet implementation reproduces the presented numerical examples.
* [ ] Calculate **α, β, G1, G2**
* [ ] Classify (ResNet)
* [ ] Save results (CSV)

### 🧪 **Experiment M1 — Wavelet Baseline (Mandatory)**

* [ ] Implement the **Wavelet-Packet** experiment in `/src/experiments/`
* [ ] Extract feature vectors (sub-band energies)
* [ ] Apply **[0,1] normalization**
* [ ] Compute paraconsistent metrics:
  * [ ] α (intra-class)
  * [ ] β (inter-class)
  * [ ] G1, G2
* [ ] Perform classification using **ResNet** (fixed configuration)
* [ ] Save results (CSV)

🎯 **Objective:** establish a deterministic theoretical baseline.

---

## 🔬 PHASE 2 — LFCC × MEL × BARK (CENTRAL)
* [x] LFCC / MEL / BARK Pipeline - Implemented in `src/experiments/01/`
  * Complete LFCC extraction with preprocessing, framing, FFT, filterbank, DCT
  * Calculation of delta and delta-delta coefficients
  * Audio loading from .mat files
  * Subject processing orchestration
* [ ] Execute **LFCC × MEL × BARK** (full comparison)
* [ ] Analyze specific impact of **LFCC vs perceptual scales** in /src/experiments/.
* [ ] **Systematically compare LFCC × MEL × BARK** - mandatory experimental step
  * Comparison in terms of paraconsistent separability (α, β, G1, G2)
  * Classification performance
  * Robustness to severe speech degradations
* [ ] Modalities:
  * [ ] Voice
  * [ ] EEG
  * [ ] Voice + EEG
* [ ] Metrics:
  * [ ] α, β, G1, G2
  * [ ] Accuracy
* [ ] Consolidate comparative table

### 🧪 **Experiment M2 — LFCC × MEL × BARK Comparison (Central)**

* [ ] Run the **LFCC** pipeline
* [ ] Run the **MEL / MFCC** pipeline
* [ ] Run the **BARK** pipeline
* [ ] Keep:
  * the same window
  * the same classifier
* [ ] Evaluate, for each feature set:
  * [ ] α, β, G1, G2
  * [ ] Accuracy
* [ ] Consolidate results into a single comparative table

🎯 **Objective:** validate the superiority/adequacy of LFCC (Chap. 2.1.12).

---

## 🧠 PHASE 3 — Autoencoders
* [x] Sub-complete AE (implemented in /src/demos/)
* [x] Supra-complete AE (implemented in /src/demos/)
* [ ] **Denoising AE** (mandatory, in /src/experiments/)
* [ ] Fix architecture and bottleneck
* [ ] Compare:
  * [ ] AE × Wavelet-Packet
  * [ ] Classical × Regularized × Denoising
  * wavelet-packet features × autoencoder features;
  * classical autoencoder × regularized × denoising.
* [ ] Evaluate (α/β + accuracy)

### 🧪 **Experiment M4 — Autoencoders (Feature Learning)**

* [ ] Implement **sub-complete autoencoder** (experiments)
* [ ] Implement **supra-complete autoencoder** (experiments)
* [ ] Implement **denoising autoencoder** (mandatory)
* [ ] Fix and document:
  * architecture
  * bottleneck size
  * stopping criterion
* [ ] Extract learned features
* [ ] Compare:
  * [ ] classical AE × regularized × denoising
  * [ ] AE × Wavelet-Packet
* [ ] Evaluate:
  * [ ] α, β, G1, G2
  * [ ] Accuracy

🎯 **Objective:** validate feature learning versus classical feature engineering.

---

## 🔀 PHASE 4 — Modalities (CLOSURE)
* [ ] Pronounced speech
* [ ] Imagined speech
* [ ] Mixed speech
* [ ] Compare:
  * [ ] Voice only
  * [ ] EEG only
  * [ ] Voice + EEG
* [ ] Cross-validation between:
  * different types of features;
  * different modalities (voice × EEG) in /src/experiments/.
* [ ] Demonstrate gain with:
  * [ ] Imagined speech
  * [ ] EEG + voice fusion
* [ ] Demonstrate gains from:
  * imagined speech;
  * EEG + voice fusion in /src/experiments/.

### 🧪 **Experiment M3 — Multimodality (EEG × Voice)**

* [ ] Select the base feature (LFCC or best from M2)
* [ ] Run experiments with:
  * [ ] Voice only
  * [ ] EEG only
  * [ ] Voice + EEG (fusion)
* [ ] Compute:
  * [ ] α, β, G1, G2
  * [ ] Accuracy
* [ ] Compare unimodal × multimodal performance

🎯 **Objective:** demonstrate gains from EEG + voice fusion.

### 🧪 **Experiment M5 — Imagined Speech (Thesis Differential)**

* [ ] Select the best-performing feature (from previous experiments)
* [ ] Run scenarios:
  * [ ] Phonated speech
  * [ ] Imagined speech
  * [ ] Mixed speech
* [ ] Compute:
  * [ ] α, β, G1, G2
  * [ ] Accuracy
* [ ] Compare the impact of imagined speech

🎯 **Objective:** fulfill the central objective of the thesis.

---

## 🛡️ PHASE 5 — Robustness (OPTIONAL)
* [ ] Insert controlled noise
* [ ] Measure drop in **G1** and accuracy
* [ ] Validate SNN tolerance
* [ ] Validate SNN noise tolerance as discussed in Chap. 2.1.9.

### 🧪 **Experiment M6 — Noise Robustness (Supportive, Optional)**

* [ ] Introduce controlled noise into the signals
* [ ] Evaluate progressive degradation of:
  * [ ] G1
  * [ ] Accuracy
* [ ] Analyze SNN noise tolerance

🎯 **Objective:** support claims regarding SNN robustness (Chap. 2.1.9).

---

## 📊 PHASE 6 — Final Consolidation
* [ ] Table **LFCC × MEL × BARK**
* [ ] Table **Wavelet × AE × Denoising**
* [ ] Graph in paraconsistent plane
* [ ] Check traceability with Section 1.1.1
* [ ] Report:
  * classical metrics (accuracy, etc.);
  * paraconsistent metrics in /src/experiments/.
* [ ] Ensure that all experiments are linked to the objectives listed in Section 1.1.1.

### 📊 **Final Consolidation (Mandatory)**

* [ ] Final table:
  * **LFCC × MEL × BARK**
* [ ] Final table:
  * **Wavelet × AE × Denoising**
* [ ] Plot in the **paraconsistent plane**
* [ ] Verify traceability:
  * experiment → objective from Section 1.1.1



### ✅ Implemented Components (/src/core/ and /src/demos/)

#### Core Neural Network Components
* [x] Implement complete **LIF** neuron.
* [x] Implement **surrogate gradient function** (Exponential / SuperSpike).
* [x] Implement **L1 and L2 regularization** (Appendix A).

#### Signal Processing and Feature Extraction
* [x] Implement **DWT** via Mallat algorithm.
* [x] Implement **Wavelet-Packet Transform (WPT)**.
* [x] Select wavelets based on frequency and phase response.
* [x] Implement **BARK** (energies per band).
* [x] Implement **MEL / MFCC** (with DCT).
* [x] **Implement LFCC** (voice and EEG) - both signals implemented.
* [x] Calculate delta and delta-delta coefficients.

#### Paraconsistent Analysis
* [x] Implement calculation of **α (intraclass similarity)**.
* [x] Implement calculation of **β (interclass overlap)**.
* [x] Calculate **G1** and **G2**.
* [x] Calculate distances to the optimal point **(1,0)** in the paraconsistent plane.
* [x] Use these values as **primary quality criterion for the vectors**.

#### Autoencoders
* [x] Implement **sub-complete autoencoder** (in /src/demos/).
* [x] Implement **supra-complete autoencoder** (in /src/demos/).

---

## Executive Summary

For the task of speaker identification using synchronized EEG and audio data, the recommended starting point is a **1.5-second window with a 50% overlap**. This duration is long enough to capture prosodic and intonational cues fundamental to speaker identity, which are often more discriminative than short-term phonetic features (Snyder et al., 2018). The 50% overlap ensures a good trade-off between temporal resolution and computational efficiency, generating a sufficient number of samples for training deep learning models without excessive redundancy.