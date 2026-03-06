# Multimodal Speaker Identification and Neural Network Framework Guide

This guide consolidates the methodology, architecture, and implementation details for a robust speaker identification pipeline using EEG and audio, leveraging Spiking Neural Networks (SNNs), autoencoders, and paraconsistent analysis. It also documents the C++20 neural network framework and practical engineering notes for reproducible research and development.

---

## 1. Methodological Foundations

### Data Acquisition and Synchronization
- Use multichannel EEG and simultaneous audio, ensuring strict temporal synchronization.

### Preprocessing and Dimensionality Reduction
- Segment signals into short windows (e.g., 100 ms, with optional 50% overlap) to capture relevant temporal dynamics.
- Downsample audio to 16 kHz and EEG to 200 Hz, preserving the informative frequency bands while reducing dimensionality.

### Input Structuring
- For each window: concatenate 1600 audio samples (100 ms @ 16 kHz) and 20 samples per EEG channel (100 ms @ 200 Hz).
- No manual feature extraction is performed before the autoencoder.

### Feature Extraction Pipelines
#### Autoencoder Path
- Dense, temporal convolutional, or variational autoencoders map input vectors to a latent space (z = f_θ(x)).
- Experiment with latent dimensions (16–128), 2–6 layers, ReLU/GELU activations, and dropout/L2 regularization.

#### Wavelet Path
- Apply wavelet transforms (Daubechies, Symlets, Coiflets, Morlet, Mexican Hat) separately to EEG and audio.
- Extract features such as energy per scale, wavelet entropy, and band variance.

### Paraconsistent Feature Engineering
- For each feature vector, compute degrees of favorable (μ) and contrary (λ) evidence using Annotated Paraconsistent Logic (LPA).
- Derive certainty (Gc = μ - λ) and contradiction (Gct = μ + λ - 1) metrics to assess class separability and feature consistency.
- Use paraconsistent distance, separability, and inconsistency indices for evaluation.

### Comparative Evaluation
- Compare autoencoder, wavelet, and combined features using paraconsistent metrics and downstream classifiers (SVM, MLP, SNN).
- Select the model with highest separability, lowest contradiction, and best generalization.

### Scientific Evidence and Limitations
- Window sizes of 50–200 ms are standard in EEG/audio studies (Cohen 2014, O'Shaughnessy).
- Downsampling to 16 kHz is common in ASR (Rabiner & Schafer 2011).
- Autoencoders and paraconsistent analysis are established in the literature (Waytowich et al. 2018, Abe 2015).
- Limitations: fixed window may miss long EEG events; autoencoders risk overfitting noise; paraconsistent metrics depend on evidence definitions; optimal wavelets are signal-dependent.

### Recommended Extensions
- Explore multimodal autoencoders with attention, contrastive learning, sparse autoencoders, and CCA/Deep CCA for aligned latent spaces.

---

## 2. Neural Network Framework (C++20)

This framework provides a high-performance, modern C++20 implementation for SNNs, autoencoders, and EEG/audio synchronization, with a focus on performance, security, and maintainability.

- Modular design: core abstractions for tensors, layers, data loaders, optimizers, statistics, and utilities.
- Spiking Neural Networks: Leaky Integrate-and-Fire (LIF) and LeakyIntegrator readout layers.
- Autoencoders: dense, convolutional, variational, and denoising variants.
- Data handling: deterministic batching, MAT/NumPy file support, and robust normalization.
- Performance: SIMD vectorization, OpenMP parallelization, and memory-efficient operations.
- Testing: Google Test integration, 95%+ coverage, static analysis (Cppcheck, Flawfinder, Clang-Tidy).
- Security: input validation, bounds checking, and RAII resource management.

### Example Usage
```cpp
#include "nn/tensor/Tensor.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/LeakyIntegrator.hpp"
// ...
```

---

## 3. System Architecture and Portability

The pipeline supports end-to-end speaker identification/verification, from audio/EEG capture through feature extraction, spike encoding, and SNN-based classification. CLI-driven flows support demo, enrollment, training, identification, verification, and evaluation. Data schemas, modular boundaries, and deterministic processing are strictly enforced for reproducibility and portability.

### Key Engineering Laws
- Always use core abstractions (tensor, layer, DataLoader, optimizer) for extensibility and testability.
- No hidden global state; propagate errors explicitly; maintain deterministic seeding and ordering.
- Vendor dependencies (Eigen, FFTW3, NFFT3, cnpy, matio, yaml-cpp, matplotlib-cpp, argparse, imgui/implot, GoogleTest) are managed via CMake and must be replaced only with equivalent, justified alternatives.

---

## 4. Experimental Pipeline and Reproducibility

### Phases
1. **Freezing & Infrastructure**: Fix window/overlap, normalization, classifier architecture, and config management.
2. **Classical Feature Engineering**: Wavelet/WPT baseline, reproducibility, and paraconsistent metrics.
3. **Spectral Scales**: Compare LFCC, MEL, BARK representations.
4. **Feature Learning**: Spiking autoencoders (sub/supra/denoising), feature extraction, and comparison.
5. **Modalities**: Unimodal (voice/EEG) vs. multimodal (voice+EEG) analysis.
6. **Imagined Speech**: Evaluate phonated, imagined, and mixed speech scenarios.
7. **Noise Robustness**: Inject noise, measure degradation, and compare clean vs. noisy signals.
8. **Final Consolidation**: Comparative tables, paraconsistent plots, and state-of-the-art benchmarking.

### Metrics
- Paraconsistent (α, β, G1, G2), accuracy, F1-score, MACs, RTF.
- Robustness, computational efficiency, and multimodal gain.

---

## 5. Advanced Topics

### LeakyIntegrator Readout Layer
The LeakyIntegrator is a continuous-valued readout for SNNs, acting as a low-pass filter on spike trains. Use it as the final decoder layer for regression or reconstruction tasks, or for debugging gradient flow.

### Multi-Pass Forward and Loss Modes
For SNNs with stochasticity (e.g., Poisson coding), aggregate outputs over multiple forward passes to reduce variance. Implement configurable loss modes (rate, Monte Carlo, temporal pooling, van Rossum, membrane, cosine, MSE vector) and always compute loss after aggregation. Ensure CLI/config compatibility and GPU safety.

---

## 6. Engineering and Maintenance

- Follow modular code structure and update documentation for new features.
- Use static analysis and coverage tools before submitting changes.
- Profile performance and memory usage regularly.
- Maintain experiment scripts, configs, and result logs for traceability.

---

## 7. References

- Cohen, M. X. (2014). *Analyzing Neural Time Series Data*
- O'Shaughnessy, D. (Speech Processing)
- Rabiner, L. R., & Schafer, R. W. (2011). *Theory and Applications of DSP*
- Waytowich, N. R., et al. (2018). Deep learning for EEG
- Hsu, W.-N., et al. (2017). Unsupervised speech representation
- Abe, J. M. (2015). *Paraconsistent Intelligent Based Systems*
- Da Costa, N. C. A. (Lógica Paraconsistente)

---

This guide is intended as a living document for both research and engineering teams. Update as the methodology, codebase, or experimental protocol evolves.
