# Signal Windowing — Plain Language Guide

> **Technical reference:** [Windowing](../Windowing.md)

---

## Why do we chop signals into windows?

A speech or EEG recording can last seconds or minutes. Neural networks and Fourier analysis work on fixed-size inputs. We split the long signal into short, overlapping segments called **windows** (or *frames*).

Each window is analysed independently as if the signal were approximately constant over that short duration. For speech at 8 kHz, a 512-sample window (64 ms) is short enough that the vocal tract shape barely changes — so the spectrum within that window meaningfully represents one "snapshot" of the voice.

---

## Overlapping windows

Adjacent windows overlap significantly. A typical setup:
- **Window length**: 512 samples (64 ms)
- **Hop size** (step between windows): 256 samples (32 ms)
- **Overlap**: 50% (256 samples shared between consecutive windows)

Why overlap?
- Features at the edge of a window are affected by the window boundary. Overlapping ensures every point in the signal is well-represented in at least one window.
- Overlapping provides more feature vectors per second, which helps temporal classifiers.

---

## The spectral leakage problem and how window functions fix it

When you apply FFT to a segment of a signal, the FFT assumes the segment repeats infinitely. If the signal doesn't connect smoothly at the edges (start ≠ end), this creates a sharp "jump" that appears as fake high-frequency content in the spectrum.

**Window functions** solve this by multiplying the frame by a smooth curve that tapers to zero at both edges:

```
Rectangular window:  1.0  1.0  1.0  1.0  1.0  (hard edges → leakage)
Hamming window:      0.08 0.54 1.0  0.54 0.08  (smooth taper → less leakage)
Hann window:         0.0  0.5  1.0  0.5  0.0   (tapers all the way to zero)
```

The smoother the taper, the less spectral leakage — but at the cost of slightly blurring nearby frequencies together.

**For speech and EEG:** Hamming or Hann windows are standard. The Blackman window is even smoother but blurs frequencies more. The rectangular window (no taper) is only appropriate when you know the signal connects smoothly at frame boundaries.

---

## Window size: the time-frequency trade-off

You cannot have perfect time resolution *and* perfect frequency resolution simultaneously. This is a fundamental limit of signal processing (the Heisenberg uncertainty principle of signals):

| Shorter window | Longer window |
|---|---|
| Better time resolution (localise events in time) | Better frequency resolution (resolve close frequencies) |
| Worse frequency resolution | Worse time resolution |
| Fewer samples → less reliable FFT | More samples → more reliable FFT |

For speech at 8 kHz:
- **512 samples (64 ms)** → good frequency resolution, can see individual formants
- **256 samples (32 ms)** → better temporal tracking of fast consonants, but coarser spectrum

For EEG at 800 Hz:
- **256 samples (320 ms)** → captures slow delta/theta waves, misses fast transitions
- **64 samples (80 ms)** → captures beta/gamma bands, but frequency resolution is coarser

---

## Windowing in this project

The `WindowingEngine` in the library takes a full signal and produces a list of windowed frames, each ready for FFT or wavelet analysis. The LFCC and MFCC pipelines call this internally.

---

## See also

- [Windowing (technical)](../Windowing.md) — window function formulas, API
- [Wave (plain)](./Wave.md) — what happens to each window after splitting
- [LFCC (plain)](../../Concepts/Plain/LFCC.md) — LFCC uses windowed frames as input
