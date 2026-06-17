# Multimodal EEG + Audio Prototype (run_prototype.py)

Python prototype for multimodal EEG and audio fusion using autoencoders and paraconsistent analysis. Loads the *BaseDeDatosHablaImaginada* corpus, windows the signals, trains a dense or spiking autoencoder to compress each modality to a shared latent space, extracts complementary DWT statistics, and evaluates three linear classifiers (latent-only, wavelet-only, combined) under the Da Costa paraconsistent logic framework. Generates a JSON summary and NPZ feature archive.

## Algorithm

### Preprocessing

**Audio** (target $f_s = 16\,\text{kHz}$): resampled from native rate via `scipy.signal.resample_poly`, then z-scored per window.

**EEG** (target $f_s = 200\,\text{Hz}$): resampled and z-scored per channel per window.

Window duration: $T_w = 100\,\text{ms}$

$$N_\text{audio} = f_s^\text{audio} \cdot T_w = 1600\,\text{samples}, \quad N_\text{EEG} = f_s^\text{EEG} \cdot T_w = 20\,\text{samples/channel}$$

Concatenated input per window: $\mathbf{x} = [\mathbf{x}_\text{audio}, \text{vec}(\mathbf{x}_\text{EEG})] \in \mathbb{R}^F$.

### Autoencoder compression

**Dense autoencoder** (`DenseAutoencoder`): symmetric encoder/decoder with GELU activations.

$$\text{Encoder: } \mathbf{z} = \text{GELU}(\mathbf{W}_L \cdots \text{GELU}(\mathbf{W}_1 \mathbf{x})), \quad \mathbf{z} \in \mathbb{R}^{d_\text{lat}}$$

Defaults: `hidden_dims=(512, 256)`, `latent_dim=64`.

**Spiking autoencoder** (`SpikingAutoencoder`): snnTorch `Leaky` neurons with learnable $\beta$:

$$\beta = e^{-\Delta t / (R \cdot C)}, \quad \Delta t = 10^{-3}\,\text{s},\; R = 5.0,\; C = 1.0,\; V_{\text{th}} = 1.0$$

Temporal mean of $T$ spike outputs forms the latent code. Custom `ExponentialSurrogate` as `torch.autograd.Function` for BPTT. Default `snn_time_steps=5`.

Training objective: MSE reconstruction loss, Adam optimiser.

### DWT feature extraction

For each signal channel, `pywt.wavedec(signal, 'db4', level=4)` returns $J+1 = 5$ subbands. Per subband, three statistics are extracted:

$$\text{energy} = \sum_n c[n]^2, \quad \text{variance} = \text{Var}(c), \quad \text{entropy} = -\sum_n c[n]^2 \log(c[n]^2 + \varepsilon)$$

Audio wavelet features: $5 \times 3 = 15$ values.
EEG wavelet features: $C \times 5 \times 3$ values (per-channel).
Combined: audio feats + EEG feats concatenated → $\mathbf{w} \in \mathbb{R}^{D_w}$.

### Classification

Three linear classifiers trained independently (no hidden layers):

1. **Latent classifier**: input $\mathbf{z} \in \mathbb{R}^{64}$
2. **Wavelet classifier**: input $\mathbf{w} \in \mathbb{R}^{D_w}$
3. **Combined classifier**: input $[\mathbf{z}, \mathbf{w}] \in \mathbb{R}^{64+D_w}$

Cross-entropy loss, Adam, train/val split as configured.

### Paraconsistent analysis (Da Costa)

Given class probability vector $\mathbf{p} \in \Delta^{C-1}$ and true label $y$:

$$\mu = p_y \quad \text{(belief in correct class)}$$
$$\lambda = \max_{c \neq y} p_c \quad \text{(belief in most competitive wrong class)}$$
$$G_c = \mu - \lambda \quad \text{(certainty degree)}$$
$$G_{ct} = \mu + \lambda - 1 \quad \text{(contradiction degree)}$$

Aggregate over the test set: $\bar{G}_c$, $\sigma_{G_c}$, $\bar{G}_{ct}$, $\sigma_{G_{ct}}$.

High $G_c > 0$ with low $|G_{ct}|$ indicates consistent, confident classification. High $G_{ct}$ indicates the model simultaneously supports two competing hypotheses — a paraconsistent state.

## Architecture

```
BaseDeDatosHablaImaginada/
  audio files + EEG files
          │
    preprocess.py
    (resample → z-score → 100 ms windows → WindowedRecord)
          │
    models.py
    (DenseAutoencoder or SpikingAutoencoder)
    train on MSE  →  encoder extracts z ∈ R^64
          │
    wavelet_features.py
    (pywt.wavedec db4 level 4 → energy/variance/entropy per subband)
          │
    concat [z, w]  →  3 linear classifiers (PyTorch)
          │
    paraconsistent.py
    (μ/λ from softmax probs → Gc, Gct per sample)
          │
    JSON summary + NPZ features archive
```

## Theory & State of the Art

Multimodal fusion of EEG and audio for imagined speech recognition is an open problem. EEG captures neural correlates of speech imagery at high temporal resolution but low spatial resolution; audio provides a ground-truth acoustic reference. Joint autoencoder compression learns a shared latent manifold that can separate speaker-specific patterns from session noise (Palazzo et al., 2020).

Spiking autoencoders (Rathi & Roy, 2020) leverage temporal sparsity to represent data efficiently on neuromorphic hardware. The `SpikingAutoencoder` here uses rate-coded output (temporal mean) to produce a continuous latent code compatible with conventional downstream classifiers.

The Da Costa paraconsistent logic framework (Da Costa, 1974; Abe & Da Silva Filho, 1992) provides a formal basis for reasoning under contradiction. In classification, $G_c$ and $G_{ct}$ separate two failure modes: low certainty (model is uncertain but not contradictory) vs. high contradiction (model simultaneously supports multiple hypotheses). This is particularly relevant for multimodal EEG data where sensor artefacts produce conflicting evidence across modalities.

DWT statistics (energy, variance, entropy) as speaker features follow the work of Polikar et al. and are related to wavelet packet energy classifiers used in voice activity detection and phoneme recognition (Vetterli & Kovacevic, 1995).

## How to Use (HOWTO)

### Requirements

```bash
pip install torch torchaudio snntorch pywavelets scipy numpy
```

### Run (full pipeline)

```bash
python src/demos/pyDemos/multimodal_eeg_audio/run_prototype.py \
    --data-root /path/to/BaseDeDatosHablaImaginada \
    --output-dir results/multimodal_prototype \
    --epochs 20 \
    --batch-size 32 \
    --device cuda
```

### Key options

```
--data-root PATH        Dataset root directory
--format mat|wav        File format selector (default: mat)
--audio-orig-sr INT     Native audio sample rate (default: 44100)
--eeg-orig-sr INT       Native EEG sample rate (default: 256)
--output-dir PATH       Where to write results
--device cpu|cuda       PyTorch device
--epochs INT            Autoencoder training epochs (default: 30)
--batch-size INT        Mini-batch size (default: 64)
--seed INT              Global RNG seed (default: 42)
--lr FLOAT              Adam learning rate (default: 1e-3)
--weight-decay FLOAT    L2 regularisation (default: 1e-4)
--no-save-features      Skip NPZ feature archive
```

### Expected Output

```
results/multimodal_prototype/
  summary.json           — classification accuracy, Gc/Gct per classifier
  features.npz           — keys: z_latent, w_wavelet, labels, speaker_ids
  training_curve.csv     — epoch vs reconstruction loss
```

`summary.json` structure:
```json
{
  "latent_accuracy": 0.72,
  "wavelet_accuracy": 0.68,
  "combined_accuracy": 0.79,
  "paraconsistent": {
    "combined": {"mean_gc": 0.41, "std_gc": 0.18, "mean_gct": -0.12, "std_gct": 0.09}
  }
}
```

## Dependencies

| Library | Purpose |
|---|---|
| `torch`, `torchaudio` | Neural network training and audio resampling |
| `snntorch` | Spiking autoencoder `Leaky` neurons and surrogate gradients |
| `pywavelets` (`pywt`) | DWT feature extraction (`wavedec`) |
| `scipy` | `resample_poly` for audio/EEG resampling |
| `numpy` | Array operations and NPZ I/O |
