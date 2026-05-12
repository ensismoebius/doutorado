# Wavelet Transform — Plain Language Guide

> **Technical reference:** [Wavelet](../Wavelet.md)

---

## Why not just use Fourier analysis?

The Fourier Transform (FFT) answers: "what frequencies are present in this signal?" It gives you the full spectrum — but throws away *when* those frequencies occurred.

For a speech recording, the Fourier transform tells you there is energy at 300 Hz, 1000 Hz, 2500 Hz. But it cannot tell you whether the 2500 Hz component came from the beginning, middle, or end of the recording.

For speaker verification and EEG analysis, timing matters. A person's voice changes over the recording. EEG patterns evolve over time. We need to know both *what frequencies* are present *and when*.

---

## The wavelet idea: a sliding frequency microscope

A wavelet is a small wave — a brief oscillation that is non-zero for a short time and zero everywhere else (it is "compact").

The wavelet transform asks: "how much does this small wave, at this particular scale (frequency) and time position, match my signal?"

By sliding this small wave across the signal at different scales (stretching it to capture low frequencies, compressing it to capture high frequencies), the transform produces a 2D picture:
- **X axis**: time position
- **Y axis**: scale (frequency)
- **Brightness**: how strongly the signal matches the wavelet there

This is the key difference from FFT: you get a time-frequency map instead of just a frequency map.

---

## The practical version: Discrete Wavelet Packet Transform (DTWPT)

In this project, the specific variant used is the **Discrete-Time Wavelet Packet Transform (DTWPT)**. It works by repeatedly splitting the signal with two filters:

1. **Low-pass filter**: keeps slow, smooth components (approximation)
2. **High-pass filter**: keeps fast, sharp components (detail)

After one level of filtering and downsampling, you get two sub-signals, each at half the original length. Apply the same process again to both → four sub-signals. Continue for several levels.

The result is a binary tree of sub-signals, each representing a different frequency band at a different time resolution:

```
Original signal (1024 samples)
├── Low (512 samples, 0–2 kHz)
│   ├── Low-Low (256 samples, 0–1 kHz)
│   └── Low-High (256 samples, 1–2 kHz)
└── High (512 samples, 2–4 kHz)
    ├── High-Low (256 samples, 2–3 kHz)
    └── High-High (256 samples, 3–4 kHz)
```

The "packet" variant processes both the low and high branches (vs. standard DWT that only recurses on the low branch). This gives more uniform frequency resolution across the spectrum.

---

## Wavelet families: choosing the right shape

Different wavelets have different shapes, and the choice affects what patterns are captured:

| Wavelet | Shape | Best for |
|---|---|---|
| **Haar** | Square step | Sharp edges, transients |
| **Daubechies-4 (Daub4)** | Smooth, 4-tap | General speech and EEG signals |
| **Daubechies-6 (Daub6)** | Smoother, 6-tap | Smoother signals |
| **Daubechies-8+** | Even smoother | Very smooth signals |

More "vanishing moments" (Daub4 has 2, Daub8 has 4...) means the wavelet is smoother and better at representing smooth signals. Haar is the simplest and fastest but captures sharp steps better than smooth curves.

---

## Energy features from wavelet packets

After the DTWPT, each sub-band contains a sequence of coefficients. For speaker/EEG classification, the next step is to compute the **energy** of each sub-band:

```
energy of sub-band k = sum of (coefficient)² for all coefficients in that sub-band
```

This produces one number per sub-band. With a 3-level DTWPT producing 8 sub-bands, you get an 8-dimensional feature vector per signal frame. These energy values are what gets passed to the paraconsistent analysis.

---

## Why DTWPT for this project?

The paraconsistent analysis compares how well different feature extraction approaches separate speaker classes. DTWPT provides a classical, interpretable baseline:
- Each sub-band energy has a clear physical meaning (energy in a specific frequency range at a specific time)
- The wavelet family and decomposition depth are tunable hyperparameters
- It is fast and deterministic — no training required

The comparison is: does the learned representation from an autoencoder beat the engineered DTWPT features? The paraconsistent D_truth metric answers this question.

---

## See also

- [Wavelet (technical)](../Wavelet.md) — transform equations, implementation, API
- [Paraconsistent (plain)](./Paraconsistent.md) — how wavelet features are quality-evaluated
- [Wave (plain)](./Wave.md) — audio preprocessing before wavelet transform
- [LFCC (plain)](../../Concepts/Plain/LFCC.md) — alternative filterbank approach
