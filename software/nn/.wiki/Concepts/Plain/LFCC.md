# LFCC — Plain Language Guide

> **Technical reference:** [LFCC](../LFCC.md)

---

## What is LFCC?

LFCC stands for **Linear Frequency Cepstral Coefficients**. It is a method to turn a raw audio recording into a compact set of numbers that describe the shape of the speaker's vocal tract.

Think of it as a fingerprint of someone's voice — not the words they said, but the physical characteristics of how their voice sounds.

---

## The problem with the most popular alternative (MFCC)

The most common feature for speech processing is MFCC (Mel Frequency Cepstral Coefficients). It was designed to model how *human hearing* works — we are more sensitive to differences in low-frequency sounds than high-frequency ones. So MFCC gives a lot of detail to low frequencies and compresses high frequencies together.

This is great for **speech recognition** (understanding words). But it's not ideal for **speaker verification** (recognising who is speaking).

Why? Because a person's unique voice characteristics are partly carried in high-frequency content:
- The fine structure of how their vocal cords vibrate (jitter, shimmer)
- Higher formants (resonances of the vocal tract above 4 kHz) that depend on larynx and pharynx shape
- Spectral details above the range that MFCC treats carefully

By compressing those high frequencies, MFCC throws away some of the information that makes your voice *yours*.

---

## What LFCC does instead

LFCC uses a **linear** (evenly-spaced) filterbank. Imagine placing equally-spaced listening buckets across the full frequency range from 0 Hz to 4000 Hz. Each bucket collects the energy of sounds in its frequency range.

Because all buckets are the same width, high frequencies get the same attention as low frequencies. Nothing is thrown away.

```
0 Hz                     4000 Hz
|-----|-----|-----|-----|-----|-----|  LFCC: equal spacing
|--------|--------|-------------|      MFCC: wider buckets at high freq
```

---

## The processing steps in plain terms

1. **Divide the signal into short overlapping windows** (typically ~64 ms each)
2. **Compute the frequency content** of each window using FFT (fast Fourier transform — converts time to frequency domain)
3. **Apply the linear filterbank**: sum up the energy in each equally-spaced band
4. **Take the logarithm** of each band's energy (this compresses the dynamic range, like our ears do for loudness)
5. **Apply DCT** (discrete cosine transform) — this decorrelates the filter outputs and packs the most information into the first few numbers
6. **Keep the first N coefficients** (typically 13) as the feature vector for that window

The result: each window of audio becomes a vector of 13 numbers that describe the spectral shape.

---

## What the final numbers mean intuitively

- **Coefficient 0** — overall energy (loudness of the frame)
- **Coefficient 1** — broad shape (is energy concentrated at low or high frequencies?)
- **Coefficients 2–12** — progressively finer spectral shape details

In speaker verification, the higher coefficients (above coefficient 4 or so) are what carry the most speaker-discriminative information because they capture the fine spectral shape differences between vocal tracts.

---

## Why 8 kHz sampling rate?

By Nyquist's theorem, to faithfully represent frequencies up to F Hz, you need to sample at least 2×F times per second. At 8 kHz (8000 samples/second), you can represent frequencies up to 4000 Hz — exactly the range used in telephone-quality speech, which is enough to capture the first few formants and vocal tract characteristics.

---

## Where LFCC fits in the thesis

```
Speech recording (8 kHz)
  → divide into 512-sample windows (64 ms)
  → compute frequency spectrum of each window
  → LFCC filterbank (24 filters, 0–4000 Hz)
  → log energy of each filter
  → DCT → 13 coefficients
  → feature vector for this window

All feature vectors from all speakers
  → paraconsistent analysis → is this feature set discriminative?
  → SNN classifier → authenticate the speaker
```

---

## See also

- [LFCC (technical)](../LFCC.md) — filter equations, DCT formula, comparison with MFCC
- [Imagined Speech and EEG (plain)](./Imagined-Speech-and-EEG.md) — the EEG companion signal
- [Research Context](../../Research-Context.md) — where this fits in the thesis
