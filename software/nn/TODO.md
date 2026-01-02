# Speaker Identification Pipeline

## Executive Summary

This project establishes a comprehensive pipeline for speaker identification using synchronized EEG and audio data, leveraging Spiking Neural Networks (SNNs) and paraconsistent analysis. We begin by freezing the methodological foundations—including a fixed 1.5-second window with 50% overlap and a residual SNN classifier—ensuring reproducibility and consistent normalization across all experiments.

We then systematically compare classical feature engineering methods (wavelets) with learned feature approaches (spiking autoencoders), evaluate multiple spectral scales (LFCC, MEL, BARK), and analyze unimodal versus multimodal (EEG + voice) performance. We also introduce a phase dedicated to imagined speech to highlight the project's unique contribution, and a robustness phase to test noise tolerance.

Finally, all experiments are consolidated into comprehensive tables and paraconsistent metric plots, comparing our results with the state of the art. This pipeline is designed to be methodologically rigorous and easily reproducible, offering a robust and innovative approach to speaker identification using brain and voice signals.

**EEG + Audio using Spiking Neural Networks (SNNs)**
📌 **Updated and Corrected Task List**

---

## 🔒 PHASE 0 — Freezing & Infrastructure (DO ONCE)

**Goal:** freeze methodological decisions and guarantee reproducibility.

- [ ] Fix window length: **1.5 s / 50% overlap**
- [ ] Fix mandatory normalization: **[0,1]**
  - prerequisite for paraconsistent analysis
- [ ] Enforce normalization in `/src/experiments/`
- [ ] Fix classifier:
  - **Residual Spiking Neural Network (ResNet SNN)**

- [ ] Freeze classifier architecture
- [ ] Create a unique `config.yaml` (single source of truth)
- [ ] Define output formats:
  - CSV (metrics and results)
  - **PyTorch-compatible format** (network architectures/weights)

---

## 🧠 PHASE 1 — Classical Feature Engineering (Wavelets)

**Goal:** establish a deterministic and theoretically grounded baseline.

### 🧪 Experiment E1 — Wavelet / Wavelet-Packet Baseline

- [ ] Implement **Wavelet-Packet Transform (WPT)**
- [ ] Validate numerical examples of the decomposition
- [ ] Ensure coefficient reproducibility
- [ ] Extract **sub-band energy features**
- [ ] Apply **[0,1] normalization**
- [ ] Compute paraconsistent metrics:
  - α (intra-class similarity)
  - β (inter-class overlap)
  - G1, G2
- [ ] Perform classification using **ResNet SNN**
- [ ] Save results (CSV)

🎯 **Expected outcome:** stable, interpretable theoretical baseline.

---

## 🔬 PHASE 2 — Spectral Scales (CENTRAL PHASE)

**Goal:** compare spectral representations while keeping all other variables fixed.

### 🧪 Experiment E2 — LFCC × MEL × BARK

- [ ] Execute complete pipelines:
  - LFCC
  - MEL / MFCC
  - BARK
- [ ] Keep fixed:
  - window
  - normalization
  - classifier
- [ ] Modalities:
  - Voice
  - EEG
  - Voice + EEG
- [ ] Metrics:
  - α, β, G1, G2
  - Accuracy
  - F1-score
- [ ] Build a **single comparative table**

🎯 **Expected outcome:** identify the most suitable spectral scale for SNN + paraconsistent analysis.

---

## 🧠 PHASE 3 — Feature Learning (Spiking Autoencoders)

> **Important correction:**
> This phase occurs **before** final comparisons and is **not optional**.

### 🧪 Experiment E3 — Spiking Autoencoders

- [ ] Consolidate architectures:
  - Sub-complete AE
  - Supra-complete AE
  - **Denoising AE** (mandatory)
- [ ] Define and document:
  - number of layers
  - bottleneck size
  - stopping criteria
- [ ] Extract learned feature vectors
- [ ] Compare:
  - Wavelet-Packet × Autoencoder features
  - Sub × Supra × Denoising AEs
- [ ] Evaluate:
  - α, β, G1, G2
  - Accuracy

🎯 **Expected outcome:** validate learned features versus manual feature engineering.

---

## 🔀 PHASE 4 — Modalities (Unimodal × Multimodal)

### Fixed definitions:

- **Unimodal:** single data source
  → voice **or** EEG
- **Multimodal:** fused sources
  → voice **+** EEG

### 🧪 Experiment E4 — Multimodality

- [ ] Select the **best feature representation** (from E2 or E3)
- [ ] Run classifications:
  - Voice only
  - EEG only
  - Voice + EEG
- [ ] Compute:
  - α, β, G1, G2
  - Accuracy, F1-score
- [ ] Compare unimodal × multimodal performance

🎯 **Expected outcome:** demonstrate information gain from EEG + voice fusion.

---

## 🧠 PHASE 5 — Imagined Speech (Thesis Differential)

### 🧪 Experiment E5 — Imagined Speech

- [ ] Use the globally best configuration
- [ ] Scenarios:
  - Phonated speech
  - Imagined speech
  - Mixed speech
- [ ] Metrics:
  - α, β, G1, G2
  - Accuracy, F1-score
- [ ] Direct comparison between scenarios

🎯 **Expected outcome:** demonstrate the biometric viability of imagined speech.

---

## 🛡️ PHASE 6 — Noise Robustness (FINAL)

### 🧪 Experiment E6 — Robustness to Noise

- [ ] Inject controlled noise into:
  - audio
  - EEG / imagined speech
- [ ] Measure progressive degradation:
  - G1
  - Accuracy
- [ ] Compare:
  - clean × noisy signals
  - AE × Denoising AE

🎯 **Expected outcome:** validate structural robustness of SNN-based models.

---

## 📊 PHASE 7 — Final Consolidation (MANDATORY)

- [ ] Table:
  - **LFCC × MEL × BARK**
- [ ] Table:
  - **Wavelet × AE × Denoising**
- [ ] Plots in the **paraconsistent plane**
- [ ] Consolidation of classical metrics
- [ ] Comparison with the **state of the art**
- [ ] Final conclusions:
  - technical
  - experimental
  - scientific

---

## 📌 Final Notes

- All phases are now:
  - logically ordered
  - non-redundant
  - methodologically consistent
- Paraconsistent analysis is correctly positioned as:
  - the **primary feature-quality criterion**
- The pipeline is ready for:
  - incremental execution
  - direct thesis writing
  - experimental auditing
