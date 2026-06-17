# SNN Speaker Demo (speaker_demo / rede_snn)

End-to-end CLI demonstrating a Spiking Neural Network (SNN) pipeline for speaker identification and verification. A 440 Hz synthetic audio signal (or a WAV file) is processed through a wavelet-based feature extraction front-end, adaptively Poisson-encoded into spike trains, and fed to a residual SNN built from Leaky Integrate-and-Fire (LIF) neurons trained via BPTT with exponential surrogate gradients.

Produces two build artefacts: `rede_snn` (shared library containing the model and feature extraction) and `speaker_demo` (CLI executable).

## Algorithm

### Feature extraction

1. Pre-emphasis: $y[n] = x[n] - 0.97\,x[n-1]$
2. Framing: window size $= 512$ samples, hop $= 256$ samples
3. Power spectrum: RFFT → $P[k] = |\text{RFFT}(\mathbf{f})|^2$
4. Linear filterbank ($M = 100$ bands): triangular filters uniformly spaced on the linear axis
5. Log energies: $E_m = \log(1 + \sum_k H_m[k]\,P[k])$
6. DCT-II → 100-dimensional LFCC frame vector
7. Feature matrix: $\mathbf{X} \in \mathbb{R}^{N_\text{frames} \times 100}$

### Adaptive Poisson encoding

Max rate computed per frame:

$$r_{\max} = \text{clamp}\!\left(\frac{r_\text{target}}{\bar{x} + \varepsilon},\; 0.02,\; 0.5\right)$$

Spike probability at each time step:

$$p_i = \text{clamp}(x_i \cdot r_{\max},\; 0,\; 1), \quad s_i[t] \sim \text{Bernoulli}(p_i)$$

with `target_spikes_per_step = 0.1`, `steps_per_window = 10`.

### LIF neuron dynamics (BPTT)

Membrane potential update:

$$V[t+1] = \beta\, V[t] + (1-\beta)\, R\, I[t] - V_{\text{th}}\, S[t]$$

$$\beta = e^{-\Delta t / (R \cdot C)}, \quad \Delta t = 0.001\,\text{s},\; R = 5.0,\; C = 1.0,\; V_{\text{th}} = 0.01$$

Surrogate gradient (exponential):

$$\frac{\partial S}{\partial V} \approx \exp(-\beta_s |V - V_{\text{th}}|), \quad \beta_s = 1.0$$

## Architecture

```
Input spikes  ∈ R^{T×B×F}  (time-major: T*B rows, F cols)
    │
Linear(F → hidden_size=100)
    │
LifBPTT (surrogate=ExponentialSurrogate(1.0))
    │
ResidualSNNBlock × 3
    ├─ Linear(hidden→hidden) → LifBPTT
    ├─ Linear(hidden→hidden) → LifBPTT
    └─ skip: z + x
    │
Linear(hidden → n_classes)
    │
LifBPTT (readout_mode=false)
    │
Output spikes → rate decode → class logits
```

`ResidualSNNBlock`: each block contains two `Linear → LifBPTT` pairs with an additive skip connection on the spike output.

## CLI Subcommands

| Subcommand | Action |
|---|---|
| `demo` | Run full pipeline on synthetic 440 Hz tone; write CSV + WAV outputs |
| `capturar` | Capture audio from microphone |
| `treinar` | Train model on previously captured data |
| `identificar` | Open-set identification: return top-K speaker IDs |
| `verificar` | Closed-set verification: accept/reject a claimed identity |
| `avaliar` | Evaluate EER and minDCF on a labelled test set |

Default hyperparameters (override via CLI flags):

```
--duracao 1.0  --taxa-amostragem 44100  --tamanho-janela 512
--tamanho-passo 256  --wavelet db4  --num-bandas 100
--passos-por-janela 10  --hidden 100  --depth -1
```

## Theory & State of the Art

Speaker recognition with SNNs is an active research area motivated by the biological plausibility and potential energy efficiency of spike-based computation on neuromorphic hardware (Mahowald & Douglas, 1991; Paugam-Moisy & Bohte, 2012).

The architecture follows the SNN residual pattern of Fang et al. (2021) — "Incorporating Learnable Membrane Time Constants to Enhance Learning of Spiking Neural Networks" — where learnable $\beta$ parameters and skip connections stabilise BPTT. Exponential surrogate gradients (Shrestha & Orchard, 2018) provide a smooth approximation of the non-differentiable Heaviside step function.

Adaptive Poisson rate encoding (rate ∝ 1/mean(x)) is a biologically motivated approach to normalise firing rates across stimuli of varying amplitude, closely related to divisive normalisation in the early auditory pathway (Carandini & Heeger, 2012).

For the feature front-end, wavelet packet decomposition (Mallat, 1989) in the `wpt_voice_biometrics` sibling demo provides richer subband energy features; this demo uses a simpler linear filterbank + DCT (LFCC) for interpretability.

## How to Use (HOWTO)

### Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build out/build/max-performance --target speaker_demo -j$(nproc)
# rede_snn shared library built as dependency automatically
```

### Run (demo mode)

```bash
./out/build/max-performance/src/demos/cppDemos/snn_speaker_demo/speaker_demo demo
```

Optional flags:

```bash
./speaker_demo demo --duracao 2.0 --hidden 128 --passos-por-janela 20
```

### Expected Output

```
demo_output_spikes.csv    — frame-by-frame spike counts per band column
demo_output_audio.wav     — copy of the synthetic input audio
```

Console prints SNN forward-pass summary (layer sizes, spike statistics).

## Dependencies

| Library | Purpose |
|---|---|
| `waveCoreLib` (project) | Pre-emphasis, framing, RFFT, filterbank, DCT |
| `wavelet` (project) | Wavelet decomposition (db4 filter) |
| `tensor` (project) | `nn::Tensor` and SNN layer interface |
| `argparse` | CLI argument parsing |
| `xtensor` | Tensor arithmetic |
| `util` (project) | Shared utilities and logging |
| `linear_algebra` (project) | BLAS-accelerated matrix ops |
