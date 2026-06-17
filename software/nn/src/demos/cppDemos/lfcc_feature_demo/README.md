# LFCC Feature Extraction Pipeline (exec_lfcc_pipeline)

Batch feature-extraction pipeline that walks the *BaseDeDatosHablaImaginada* corpus, pairs per-subject Audio and EEG MAT files, and produces LFCC (Linear Frequency Cepstral Coefficient) feature matrices stored as NumPy `.npy` archives. The pipeline is the upstream feature stage for downstream SNN/ResNet classifiers in this thesis.

## Algorithm

### Pre-emphasis

High-pass shelving filter boosts high frequencies before framing:

$$y[n] = x[n] - \alpha \, x[n-1], \quad \alpha = 0.97$$

### Framing and windowing

The pre-emphasised signal is segmented into overlapping frames:

- Frame duration: $\Delta_f = 25\,\text{ms}$ → $N = \Delta_f \cdot f_s = 400$ samples at $f_s = 16\,\text{kHz}$
- Frame shift: $\Delta_s = 10\,\text{ms}$ → 160 samples
- Hamming window: $w[n] = \alpha_H - \beta_H \cos\!\left(\tfrac{2\pi n}{N-1}\right)$, with $\alpha_H = 0.54$, $\beta_H = 0.46$

### Power spectrum (RFFT)

For each windowed frame $\mathbf{f}$, the one-sided power spectrum is

$$P[k] = \left|\text{RFFT}(\mathbf{f})\right|^2, \quad k = 0, \ldots, \lfloor N/2 \rfloor$$

### Linear filterbank

$M = 24$ triangular filters equally spaced on the linear frequency axis (not the mel scale). Filter energies:

$$E_m = \sum_k H_m[k]\, P[k], \quad m = 1, \ldots, M$$

where $H_m$ is the $m$-th triangular filter.

### DCT-II (cepstral coefficients)

$$c_n = \sqrt{\frac{2}{M}} \sum_{m=0}^{M-1} E_m \cos\!\left(\pi n \left(m + 0.5\right) / M\right), \quad n = 1, \ldots, N_c$$

with $N_c = 19$ coefficients retained (DC term discarded).

### Delta features

Velocity ($\Delta$) and acceleration ($\Delta\Delta$) cepstra are computed via the regression formula (window span $\delta = 2$):

$$d_n[t] = \frac{\sum_{\tau=1}^{\delta} \tau\left(c_n[t+\tau] - c_n[t-\tau]\right)}{2\sum_{\tau=1}^{\delta}\tau^2}$$

Final feature vector per frame: $[ \mathbf{c},\, \boldsymbol{\Delta},\, \boldsymbol{\Delta\Delta} ] \in \mathbb{R}^{3 \times 19 = 57}$.

## Architecture

```
BaseDeDatosHablaImaginada/
  S<id>_Audio.mat  +  S<id>_EEG.mat
           |
           v
    process_subject()          (lfcc_pipeline_utils)
           |
    pre_emphasis (α=0.97)
    frame + Hamming window (25 ms / 10 ms shift)
    rfft_power  →  M=24 linear filters  →  log energies
    DCT-II  →  N_c=19 coefficients
    Δ + ΔΔ (span=2)
           |
    cnpy::npz_save("subject_<id>_lfcc.npz")
```

Entry point `lfcc_pipeline.cpp` iterates over subject directories with `std::filesystem::directory_iterator`, identifies pairs by suffix (`_Audio.mat` / `_EEG.mat`), and calls `process_subject` for each. EEG channels are carried alongside audio but the LFCC features are extracted from audio only; EEG windows are saved in the same NPZ for alignment.

## Theory & State of the Art

LFCCs are a variant of the MFCC (Mel Frequency Cepstral Coefficient) family (Davis & Mermelstein, 1980) that use a uniform linear filter spacing instead of the mel scale. In speaker recognition and imagined-speech EEG studies, LFCCs have been shown to preserve fine spectral structure at higher frequencies better than mel-spaced banks, which can be advantageous for short-duration utterances and for data with narrow-band artefacts.

The delta and delta-delta coefficients capture temporal dynamics within the cepstral trajectory, a standard technique originating with Furui (1986). Together, static + delta + acceleration features form the canonical 57-dimensional feature vector used in HMM-based and neural network speech frontends (Young et al., HTK Book).

For imagined speech EEG, LFCC-like features have been used as a signal-processing sanity check and as surrogate ground-truth labels: the audio LFCC trajectory defines the target class, while EEG is the actual input. This aligns with the experimental setup of the *BaseDeDatosHablaImaginada* dataset (López-Bernal et al., 2022).

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target exec_lfcc_pipeline -j$(nproc)
```

### Run

```bash
./out/build/max-performance/src/demos/cppDemos/lfcc_feature_demo/exec_lfcc_pipeline
```

The binary expects the dataset at the hard-coded relative path `BaseDeDatosHablaImaginada/` from the working directory. Adjust the path constant in `lfcc_pipeline.cpp` or invoke from the correct working directory.

### Run tests

```bash
cmake --build out/build/max-performance --target lfcc_pipeline_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R lfcc_pipeline --output-on-failure
```

Tests in `tests/lfcc_pipeline_utils_gtest.cpp` cover each pipeline stage independently: `PreEmphasisInplace`, `FramingAndWindow`, `RFFTPower`, `BuildLinearFilterbank`, `DotPowerFilterbank`, `DCT2`, `ComputeDeltas`.

### Expected Output

One `.npz` file per subject in the working directory:

```
subject_S01_lfcc.npz   →  keys: lfcc_features (F×57 float32), eeg_windows (F×C×W float32)
subject_S02_lfcc.npz
...
```

## Dependencies

| Library | Purpose |
|---|---|
| `waveCoreLib` (project) | Pre-emphasis, framing, windowing, RFFT, filterbank, DCT, deltas |
| `dataLoaders_10_1117` (project) | Loads `BaseDeDatosHablaImaginada` MAT pairs |
| `matioCpp` | Low-level MAT-5 file I/O |
| `fftw3f` (FFTW single-precision) | Real FFT backend |
| `cnpy` | NumPy `.npy`/`.npz` serialisation |
| `xtensor`, `xtensor-blas` | Tensor arithmetic |
| `tensor`, `util` (project) | Shared types and utilities |
| OpenMP | Parallel frame processing |
