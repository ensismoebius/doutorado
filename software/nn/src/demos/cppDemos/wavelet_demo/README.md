# Wavelet Demo (wavelet_demo)

Demonstrates both Discrete Wavelet Transform (DWT) and Discrete Wavelet Packet Transform (DWPT) applied to a synthetic two-component signal, using the Daubechies-8 (Daub8) wavelet. Decomposes the signal across levels 1–4 and saves time-frequency PNG plots for each level and transform type, providing a visual reference for filter bank behaviour and frequency localisation.

## Algorithm

### Signal generation

Composite test signal of length $N = f_s \cdot T$ at $f_s = 1024\,\text{Hz}$, $T = 1\,\text{s}$:

$$x[n] = 0.7\sin(2\pi \cdot 50 \cdot n / f_s) + 0.3\sin(2\pi \cdot 120 \cdot n / f_s)$$

### Discrete Wavelet Transform (DWT — Mallat algorithm)

At each level $j$, the signal is convolved with the low-pass scaling filter $h[k]$ and the high-pass wavelet filter $g[k]$, then downsampled by 2:

$$a_j[n] = \sum_k h[k]\, a_{j-1}[2n - k], \qquad d_j[n] = \sum_k g[k]\, a_{j-1}[2n - k]$$

The DWT retains only the approximation branch ($a_j$) for further decomposition, producing one detail subband per level and a final approximation. Invoked via:

```cpp
wavelets::malat(signal, filter, REGULAR_WAVELET, level)
```

### Discrete Wavelet Packet Transform (DWPT)

The DWPT extends the DWT by decomposing both approximation and detail subbands at each level, yielding a full binary tree of $2^j$ subbands at level $j$:

$$\mathbf{W}_{j,p}^{(0)}[n] = \sum_k h[k]\, \mathbf{W}_{j-1,p}[2n-k], \qquad \mathbf{W}_{j,p}^{(1)}[n] = \sum_k g[k]\, \mathbf{W}_{j-1,p}[2n-k]$$

Invoked via:

```cpp
wavelets::malat(signal, filter, PACKET_WAVELET, level)
```

At level 4, DWPT produces $2^4 = 16$ subbands each of length $N / 16 = 64$ samples.

### Filter

Daub8 (Daubechies 8-tap orthogonal wavelet) retrieved at compile time:

```cpp
auto filter = wavelets::get_wavelet<wavelets::Daub8>();
```

Daub8 has 8 vanishing moments, providing good frequency selectivity while remaining compact in the time domain.

## Architecture

```
generate_signal(f_s=1024, T=1, 50 Hz + 120 Hz)
         │
         ├──── DWT levels 1–4
         │       wavelets::malat(signal, Daub8, REGULAR_WAVELET, level)
         │       → MatplotlibCpp::savefig("wavelet_demo_dwt_N.png")
         │
         └──── DWPT levels 1–4
                 wavelets::malat(signal, Daub8, PACKET_WAVELET, level)
                 → MatplotlibCpp::savefig("wavelet_demo_dwpt_N.png")

also saves: wavelet_demo_signal.png  (original waveform)
```

Total output: 1 signal PNG + 4 DWT PNGs + 4 DWPT PNGs = 9 files.

## Theory & State of the Art

Wavelet analysis provides simultaneous time-frequency localisation, unlike the STFT which has a fixed Heisenberg time-frequency tile. The Mallat fast wavelet algorithm (Mallat, 1989) computes the multi-resolution analysis (MRA) in $O(N)$ operations via iterated filterbank stages, making it competitive with FFT-based methods.

Daubechies wavelets (Daubechies, 1988) are the canonical family of compactly supported orthogonal wavelets. The Daub8 (4th-order Daubechies) provides 4 vanishing moments with an 8-tap filter, achieving a good balance between frequency selectivity and temporal smoothness. Higher-order Daubechies filters (Daub10, Daub12) give sharper frequency roll-off at the cost of longer impulse responses.

The Wavelet Packet Transform (Coifman & Wickerhauser, 1992) extends the MRA to a best-basis selection framework: by comparing entropy or energy across the full binary tree, one can choose the subband partition that best represents a given signal class. In the speaker recognition context (`wpt_voice_biometrics` demo), WPT subband energies capture speaker-discriminative spectral fine structure that static filterbanks miss.

For audio signals, level 4 WPT at $f_s = 1024\,\text{Hz}$ gives 16 subbands of bandwidth $\approx 32\,\text{Hz}$, resolving the 50 Hz and 120 Hz components into separate subbands.

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target wavelet_demo -j$(nproc)
```

### Run

```bash
./out/build/max-performance/src/demos/cppDemos/wavelet_demo/wavelet_demo
```

No arguments. All parameters are compile-time constants.

### Expected Output

Nine PNG files written to the working directory:

```
wavelet_demo_signal.png      — original 1-second test signal
wavelet_demo_dwt_1.png       — DWT level-1 approximation + detail
wavelet_demo_dwt_2.png
wavelet_demo_dwt_3.png
wavelet_demo_dwt_4.png
wavelet_demo_dwpt_1.png      — DWPT level-1 (2 subbands)
wavelet_demo_dwpt_2.png      — DWPT level-2 (4 subbands)
wavelet_demo_dwpt_3.png      — DWPT level-3 (8 subbands)
wavelet_demo_dwpt_4.png      — DWPT level-4 (16 subbands)
```

Each DWT/DWPT plot shows subband magnitude versus sample index. The 50 Hz and 120 Hz components are visibly isolated into separate subbands from level 3 onwards.

## Dependencies

| Library | Purpose |
|---|---|
| `wavelet` (project) | `wavelets::malat`, `wavelets::get_wavelet<Daub8>`, Haar |
| `MatplotlibCpp` | PNG plot generation |
| `xtensor` | Tensor arithmetic |
| `matioCpp` | MAT file I/O (linked; not exercised here) |
| `linear_algebra` (project) | BLAS-accelerated operations |
| `serialization` (project) | Model serialisation utilities |
