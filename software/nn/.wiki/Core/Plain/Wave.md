# Audio Signal Processing — Plain Language Guide

> **Technical reference:** [Wave](../Wave.md)

---

## What is a digital audio signal?

A microphone converts air pressure changes into an electrical signal. An analog-to-digital converter (ADC) samples this signal thousands of times per second and records each measurement as a number.

For speech signals in this project, the sample rate is **8000 Hz** — 8000 measurements per second. At 16 bits per sample, one second of audio is 16,000 bytes.

The resulting signal is just an array of numbers, for example:
```
[0.002, 0.015, 0.031, 0.027, -0.003, -0.019, -0.034, ...]
```
Each number represents the air pressure at that instant.

---

## The problem: raw samples are not useful features

A raw audio array is too long and too noisy to use directly as input to a classifier. A 1-second recording at 8 kHz = 8000 numbers. Most of those numbers are correlated with their neighbors and don't carry independent information.

Feature extraction converts the raw signal into a compact, informative representation — a much smaller set of numbers that captures the shape of the speaker's voice.

---

## Step 1: divide into short overlapping windows

Speech changes slowly compared to the sample rate. A window of 512 samples (64 ms at 8 kHz) is short enough to be considered stationary (approximately constant) but long enough to capture pitch cycles.

The windows overlap: with a hop of 256 samples, each new window shifts by 32 ms, keeping half the samples from the previous window. This overlap-add structure ensures smooth feature extraction.

```
Window 1: samples [0   ... 511 ]
Window 2: samples [256 ... 767 ]
Window 3: samples [512 ... 1023]
...
```

---

## Step 2: window function

At the edges of each window, the signal is multiplied by a smooth tapering curve (Hamming or Hann window). This prevents a mathematical artifact called *spectral leakage*.

Without windowing, the FFT "sees" an abrupt discontinuity at the frame edge (the signal jumps from the last sample to zero) and creates fake high-frequency components.

The Hamming window tapers smoothly to zero at both edges, eliminating this problem.

---

## Step 3: FFT — converting from time to frequency

The **Fast Fourier Transform (FFT)** converts the windowed frame from a list of time-domain samples into a list of frequency-domain amplitudes.

Think of it like pressing the "spectrum analyser" button on a graphic equaliser. Instead of "volume over time", you see "volume by frequency".

The output is a list of complex numbers — one per frequency bin. Usually we take the absolute value squared to get the **power spectrum**: how much energy is at each frequency.

---

## Step 4: filterbank — grouping frequencies into bands

The power spectrum has hundreds of frequency bins (256 for a 512-sample FFT). Most consecutive bins are highly correlated (the spectrum is smooth). We compress them by summing energy into broader bands using a **filterbank**.

Two main filterbank types:
- **MFCC filterbank (Mel scale)**: wider bands at high frequencies, matching human hearing
- **LFCC filterbank (Linear scale)**: equally-wide bands across all frequencies, preserving high-frequency detail

See [LFCC (plain)](../../Concepts/Plain/LFCC.md) for why the linear scale is preferred for speaker verification.

---

## Step 5: log energy and DCT

Taking the **logarithm** of each band's energy:
1. Compresses the dynamic range (very large and very small values become more comparable)
2. Approximately matches how humans perceive loudness (perceived loudness is logarithmic)

Applying the **DCT** (Discrete Cosine Transform) after the log:
- Decorrelates the filter bank outputs (adjacent bands are correlated because frequencies overlap)
- Packs most information into the first few coefficients
- Lets us keep just 13 coefficients out of 24, discarding the redundant tail

---

## Pre-emphasis

Optional but common: before windowing, apply `y[n] = x[n] - 0.97·x[n-1]`. This boosts high-frequency content to compensate for the natural roll-off of the vocal tract and lips. For speaker verification with LFCC it improves the signal-to-noise ratio at high frequencies.

---

## WAV file format

The signals are stored as WAV files — a standard format for raw, uncompressed audio. The library's `Wav` class reads 16-bit PCM mono WAV files directly into a float array, normalising values to the range [-1, +1].

---

## See also

- [Wave (technical)](../Wave.md) — MFCC formulas, filterbank API, WAV loader code
- [LFCC (plain)](../../Concepts/Plain/LFCC.md) — linear filterbank specifically
- [Windowing (plain)](./Windowing.md) — the frame-splitting step
- [Wavelet (plain)](./Wavelet.md) — alternative time-frequency analysis
