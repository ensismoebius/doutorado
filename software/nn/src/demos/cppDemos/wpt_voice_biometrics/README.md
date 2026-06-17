# WPT Voice Biometrics (voice_biometrics_cpp)

Full C++ voice biometrics pipeline: loads a WAV file (or generates a synthetic 440 Hz tone), applies Wavelet Packet Transform (WPT) subband energy extraction, adaptively Poisson-encodes the energy features into spike trains, and runs them through a residual SNN for speaker feature extraction. Outputs a CSV of spike patterns per frame. Serves as the C++ counterpart to `pyDemos/snn_hyperparam_search`.

## Algorithm

### Signal windowing

Input audio is segmented into overlapping Hann-windowed frames:

$$w[n] = 0.5\left(1 - \cos\!\left(\frac{2\pi n}{N-1}\right)\right), \quad N = 512\,\text{samples},\; \text{hop} = 256$$

### Wavelet Packet Decomposition (Haar)

Each frame is decomposed with Haar WPT (`PACKET_WAVELET` mode). Decomposition level $J$ is chosen automatically:

$$J = \max\!\left(\lfloor\log_2(N_\text{window})\rfloor,\, \lceil\log_2(N_\text{bands})\rceil\right)$$

This ensures at least $N_\text{bands}$ terminal nodes. The Haar wavelet is used for its perfect reconstruction, zero phase, and computational simplicity.

### Subband energy extraction

For each of the $2^J$ terminal WPT nodes:

$$E_b = \frac{1}{|s_b|} \sum_n s_b[n]^2$$

The $2^J$ energies are linearly interpolated to exactly $N_\text{bands} = 100$ bands.

### Log normalisation

$$\tilde{E}_b = \log(1 + E_b), \quad \hat{E}_b = \frac{\tilde{E}_b - \min(\tilde{\mathbf{E}})}{\max(\tilde{\mathbf{E}}) - \min(\tilde{\mathbf{E}})}$$

### Adaptive Poisson encoding

Identical to `snn_speaker_demo`:

$$r_{\max} = \text{clamp}\!\left(\frac{r_\text{target}}{\bar{E} + \varepsilon},\; 0.02,\; 0.5\right), \quad s_b[t] \sim \text{Bernoulli}(\text{clamp}(\hat{E}_b \cdot r_{\max}, 0, 1))$$

with `target_spikes_per_step = 0.1`, `steps_per_window = 10` (default).

### SNN inference

The encoded spike tensor $\mathbf{S} \in \{0,1\}^{T \times B \times F}$ is passed through the SNN in time-major layout $(T \cdot B, F)$.

## Architecture

```
WAV / synthetic 440 Hz
        │
Hann window (512 samples, hop 256)
        │
Haar WPT  →  subband energies (2^J subbands)
        │
interpolate → N_bands=100  →  log1p + normalize [0,1]
        │
Adaptive Poisson encode  (T=10 steps/window)
        │
  Spikes ∈ {0,1}^{T×B×100}
        │
Linear(100 → hidden)  →  LifBPTT
        │
ResidualSnnBlock × depth        (depth=-1 → auto)
    ├─ Linear(hidden→hidden) → LifBPTT
    ├─ Linear(hidden→hidden) → LifBPTT
    └─ skip: z + x
        │
Linear(hidden → n_classes)  →  LifBPTT (readout)
        │
Output CSV (frame × spike_counts_per_band)
```

`depth = -1` sets the number of residual blocks to `ceil(log2(hidden_size))` automatically.

## CLI

```
voice_biometrics_cpp [options]

Options:
  --entrada-wav PATH           Input WAV file (omit for synthetic 440 Hz tone)
  --saida-csv PATH             Output CSV path
  --duracao FLOAT              Duration in seconds (default 1.0)
  --taxa-amostragem INT        Sample rate (default 44100)
  --tamanho-janela INT         Window size in samples (default 512)
  --tamanho-passo INT          Hop size in samples (default 256)
  --num-bandas INT             Number of WPT bands (default 100)
  --passos-por-janela INT      SNN time steps per window (default 10)
  --profundidade INT           Number of residual blocks (-1=auto, default -1)
  --hidden INT                 Hidden layer width (default 100)
  --seed INT                   RNG seed for Poisson encoder (default 42)
```

## Theory & State of the Art

The Wavelet Packet Transform (Coifman & Wickerhauser, 1992) provides a richer decomposition than the DWT by recursively splitting both approximation and detail branches, yielding $2^J$ frequency-uniform subbands. For speaker recognition, WPT subband energies capture glottal pulse harmonics and formant structure at finer resolution than mel filterbanks (Rao & Prasad, 2013).

The Haar wavelet (Haar, 1910) is the simplest compactly supported wavelet: $h = [1,1]/\sqrt{2}$, $g = [1,-1]/\sqrt{2}$. Despite its simplicity, Haar WPT achieves competitive performance in audio classification because subband energies are insensitive to the exact wavelet shape and more dependent on the subband frequency boundaries.

The combination of WPT features with SNN processing is directly motivated by the neuromorphic speaker recognition literature (Anwani & Rajendran, 2020; Fang et al., 2021), where event-driven spike representations reduce energy consumption on hardware like Intel Loihi or IBM TrueNorth by 1–2 orders of magnitude compared to dense neural networks.

The Poisson encoding with adaptive rate normalisation ensures that speakers with different vocal amplitudes produce similar mean spike counts, preventing the SNN from using global energy as a shortcut feature.

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target voice_biometrics_cpp -j$(nproc)
```

### Run (synthetic signal)

```bash
./out/build/max-performance/src/demos/cppDemos/wpt_voice_biometrics/voice_biometrics_cpp \
    --saida-csv output.csv
```

### Run (WAV file)

```bash
./out/build/max-performance/src/demos/cppDemos/wpt_voice_biometrics/voice_biometrics_cpp \
    --entrada-wav speaker01.wav \
    --saida-csv features_s01.csv \
    --num-bandas 100 \
    --passos-por-janela 10 \
    --hidden 128
```

### Expected Output

`output.csv` — rows = frames, columns = `frame_idx`, `band_0`, ..., `band_{N-1}`:

```
frame_idx,band_0,band_1,...,band_99
0,0.12,0.08,...,0.31
1,...
```

Console prints: frame count, WPT level chosen, SNN architecture summary.

## Dependencies

| Library | Purpose |
|---|---|
| `wavelet` (project) | Haar WPT decomposition, `PACKET_WAVELET` mode |
| `waveCoreLib` (project) | Hann windowing, interpolation, log normalisation |
| `tensor` (project) | SNN layers (`LifBPTT`, `Linear`, residual blocks) |
| `layers` (project) | `ResidualBlock` and layer registry |
| `argparse` | CLI argument parsing |
| `xtensor`, `xtensor-blas` | Tensor arithmetic |
| `util` (project) | Logging and utilities |
| `linear_algebra` (project) | BLAS-accelerated matrix ops |
| `codificacao.cpp` (from `snn_speaker_demo`) | Adaptive Poisson encoding |
