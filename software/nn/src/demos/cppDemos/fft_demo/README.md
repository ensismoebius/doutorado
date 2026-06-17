# FFT Demo (fftw3_demo)

A minimal sanity-check and dependency-verification demo that generates a synthetic two-component sinusoidal signal, computes its real-to-complex DFT via FFTW3, converts the spectral magnitudes to decibels, and plots both the time-domain waveform and the frequency-domain spectrum using matplotlib-cpp.

## Algorithm

### Signal generation

A composite signal $x[n]$ of length $N = f_s \cdot T$ is constructed as

$$x[n] = 0.7\sin\!\left(\frac{2\pi f_1 n}{f_s}\right) + 0.3\sin\!\left(\frac{2\pi f_2 n}{f_s}\right)$$

with $f_1 = 10\,\text{Hz}$, $f_2 = 40\,\text{Hz}$, $f_s = 1024\,\text{Hz}$, $T = 4\,\text{s}$.

### DFT (real-to-complex)

FFTW3 computes the one-sided spectrum $X[k]$ for $k = 0, \ldots, \lfloor N/2 \rfloor$ via an in-place `FFTW_ESTIMATE` plan:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}$$

### Magnitude in dB

Each bin magnitude is converted to the dB scale with an epsilon guard to avoid $\log(0)$:

$$M[k] = 20\log_{10}\!\left(\sqrt{\text{Re}(X[k])^2 + \text{Im}(X[k])^2} + \varepsilon\right), \quad \varepsilon = 10^{-300}$$

## Architecture

| Component | Role |
|---|---|
| `generateSignal` | Allocates FFTW-aligned memory via `fftw_malloc`; fills with the composite sine |
| `executeFFT` | Creates and executes `fftw_plan_dft_r2c_1d`; owns output buffer via `unique_ptr<fftw_complex, FFTWFreeDeleter>` |
| `calculateFFTMagnitude` | Iterates over the $\lfloor N/2\rfloor + 1$ complex bins; returns an `nn::Tensor` |
| `plotSignal` | Wraps `matplotlibcpp::plot` + `title` + `show` |
| `FFTWFreeDeleter` | RAII custom deleter for `fftw_malloc`-allocated blocks |

Data flow: `fftw_malloc` → `generateSignal` → `executeFFT` → `calculateFFTMagnitude` → `nn::Tensor` → `plotSignal`.

## Theory & State of the Art

The Cooley-Tukey FFT algorithm (Cooley & Tukey, 1965) reduces the DFT from $O(N^2)$ to $O(N \log N)$ by recursive decomposition into even and odd sub-problems. FFTW3 (Frigo & Johnson, 2005) auto-tunes the decomposition plan at runtime to near-optimal performance on the target hardware; it is the de-facto standard library for high-performance FFT in C/C++.

This demo targets the one-sided real-input spectrum (r2c), which is sufficient when the signal is real-valued; the redundant negative-frequency half is discarded. In the context of audio feature extraction, an FFT of this form is the inner kernel of MFCC/LFCC pipelines (Davis & Mermelstein, 1980) and short-time Fourier transform (STFT) spectrograms.

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target fftw3_demo -j$(nproc)
```

### Run

```bash
./out/build/max-performance/src/demos/cppDemos/fft_demo/fftw3_demo
```

No arguments. The binary uses hard-coded signal parameters ($f_s = 1024$, $f_1 = 10$, $f_2 = 40$, $T = 4\,\text{s}$).

### Expected Output

Two matplotlib windows appear sequentially:
1. **Input Signal** — time-domain plot of the 4096-sample composite sinusoid.
2. **FFT Magnitude** — one-sided dB spectrum with peaks at 10 Hz and 40 Hz.

The second window is blocking (`show(true)`); closing it terminates the process.

## Dependencies

| Library | Purpose |
|---|---|
| `FFTW3` (`fftw3`) | Real-to-complex DFT computation |
| `MatplotlibCpp` | Plot rendering via embedded Python/matplotlib |
| `xtensor`, `xtensor-blas` | Tensor backend |
| `matioCpp` | MAT file I/O (linked but not exercised in this demo) |
| `tensor` (project) | `nn::Tensor` wrapper |
| `util` (project) | Shared utilities |
