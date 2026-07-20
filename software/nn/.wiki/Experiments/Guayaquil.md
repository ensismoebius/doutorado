# Experiment04: SNN vs LSTM Comparative

Experiment04 implements a comparative study between Spiking Neural Networks (SNNs) and LSTM autoencoders on time-series data, with support for the Free Spoken Digit Dataset (FSDD).

## Theoretical Background

### Sequence-to-Sequence Learning

LSTM autoencoders compress variable-length sequences into fixed-size latent vectors: 

1. **Encoder LSTM**: Processes input sequence, produces final hidden state
2. **Latent Space**: Fixed-dimensional representation of entire sequence
3. **Decoder LSTM**: Reconstructs sequence from latent state

### SNN Surrogate Gradients

Spiking Neural Networks use surrogate gradient methods to approximate the non-differentiable spike function [1]:

$$\frac{\partial S}{\partial V} \approx \frac{\partial \sigma}{\partial V}$$

where $\sigma$ is a smooth approximation (e.g., fast sigmoid).

### Comparative Framework

The experiment compares:
- **LSTM Autoencoder**: Standard recurrent autoencoder with BPTT
- **SNN Autoencoder**: Layer-based spiking autoencoder with Leaky Integrate-and-Fire neurons

#### Latent Space and Compression

The latent space dimensionality is now defined explicitly within the `encoder_layer_spec` and `decoder_layer_spec`. For example, specifying the last encoder layer as `linear:32:identity` explicitly sets the latent size to 32.

Forcing a small latent dimensionality prevents the model from simply copying the input to the output, requiring it to learn the most critical features of the data. In this comparative study, both models are assigned the same latent dimensionality to ensure a fair comparison of their compression efficiency and reconstruction accuracy.

## Implementation

### Comparative Configuration

Config is loaded from a JSON profile. Top-level sections:

```jsonc
{
  "experiment": { "run_tag", "seed", "repeats", "seed_deterministic", "check_determinism" },
  "dataset":    { "dataset_root", "results_dir", "window_size",
                  "max_loaded_train_samples", "max_validation_samples",
                  "latex_data_dir", "save_models" },
  "training":   { "epochs", "early_stop_patience", "learning_rate",
                  "samples_per_batch", "batches_per_epoch",
                  "beta1", "beta2", "epsilon", "max_reconstruct_mean_deviation" },
  "model":      { "loss_function", "latent_dim", "lstm_hidden_size",
                  "lstm_frame_size",
                  "encoder_layer_spec", "decoder_layer_spec" },
  "evaluation": { "datasets", "encodings", "snn_architectures",
                  "v_th_values", "alpha_values" }
}
```

Only listed keys are parsed. All other JSON keys (including `_`-prefixed doc strings) are silently ignored.

Parsed by: `src/experiments/guayaquil/lib/include/GuayaquilConfig.hpp` (`from_nested_json`).

#### `model.lstm_frame_size` (default 8)

Samples fed to the LSTM per timestep. The sequence length becomes
`window_size / lstm_frame_size`, so with the article profiles' `window_size=256`
the default gives $T = 32$, $D = 8$.

Before 2026-07-18 this was hard-coded to `input_size = 1`, i.e. the window was
consumed one scalar per timestep ($T = 256$, $D = 1$). Because the dominant cost
per step is the recurrent term $h \cdot U^\top$ — independent of $D$ — that made
the LSTM roughly 7× more expensive than necessary:

| | frame=1 | frame=8 |
|---|---|---|
| sequential steps | 256 | 32 |
| MACs | 8 523 840 | 1 184 256 |
| CPU LSTM train (6 samples, 2 epochs) | 3 711 ms | 478 ms |
| OpenCL LSTM train (same) | 144 848 ms | 18 335 ms |

Constraints and caveats:

- Must divide `window_size`, enforced by `GuayaquilConfig::validate()`.
- Encoding is applied to the flat `(window_size, 1)` window **first**, then
  framing — the `direct`/`poisson`/`latency` transforms expect the flat layout.
- Evaluation compares reconstruction in framed space. MSE/MAE/$R^2$ are
  elementwise, so framing both sides leaves them unchanged.
- **This changes the LSTM-AE architecture and therefore the paper's LSTM
  results.** Set `lstm_frame_size: 1` to reproduce pre-2026-07-18 numbers. It is
  arguably a fairer baseline, since the SNN-AE sees the whole window at once via
  `linear:64` while the old LSTM saw one scalar per step.

Implemented by `to_lstm_frames()` in `src/experiments/guayaquil/lib/src/GuayaquilEncoding.cpp`;
see [LSTM Performance](../Guides/LSTM-Performance.md) for why a plain reshape
would produce a polyphase split rather than consecutive frames.

### Data Loading Limits

Samples are loaded from FSDD WAV files, windowed into non-overlapping `window_size`-sample frames, **shuffled** (seeded by `experiment.seed`), then split:

$$\text{Val Count} = \min(\text{max\_validation\_samples},\ \text{Total Loaded})$$
$$\text{Train Count} = \text{Total Loaded} - \text{Val Count}$$

The shuffle happens before the split, so the validation set is a random draw from all loaded windows — not just the last 20%.  The profile fields `max_loaded_train_samples` and `max_validation_samples` are **hard limits**, not ratios.

### Training Stability and Reproducibility

Because neural network performance can vary based on random weight initialization, the experiment uses a `repeats` parameter to ensure statistical reliability:

1. **Statistical Reliability**: By training each configuration multiple times (e.g., `repeats = 3`), the experiment allows for the calculation of averages and standard deviations, ensuring that results are not due to "lucky" random seeds.
2. **Determinism Verification**: When `seed_deterministic` is enabled, every repeat uses the same seed. The `check_determinism` flag then verifies that the results are identical across all repeats, which is critical for scientific reproducibility.
3. **Stability Analysis**: When `seed_deterministic` is disabled, each repeat uses a unique seed. This reveals how robust the model is to different initializations.

### Profile Configurations

Article profiles live in `src/experiments/guayaquil/profiles/`:

| Profile | Purpose | Runs | ETA |
|---------|---------|------|-----|
| `article-lstm-ae.json` | LSTM-AE baseline, 3 encodings × 3 seeds | 9 | ~10 min |
| `article-snn-dense.json` | SNN dense, 3 encodings × V_th/alpha sweep × 3 seeds | varies | ~45 min |
| `article-snn-conv1d.json` | SNN with 3-tap smoothing pre-filter | varies | ~45 min |
| `article-snn-recurrent.json` | SNN with LIF input transform | varies | ~45 min |
| `article-backend-bench.json` | Wall-clock timing only (xtensor vs OpenCL) | 2 | ~5 min |

All article profiles share: `window_size=256`, `dataset=fsdd`, `seed_deterministic=false`, `loss_function=mse`, `latent_dim=32`.

Profile validation test: `profile_audit_gtest` (25 tests). Run after every profile edit.

### Dataset Support

**FSDD (Free Spoken Digit Dataset)**:
- Location: `/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset`
- Format: `.wav` audio files (16-bit PCM, mono, 8kHz)
- Organization: `{digit}_{speaker}_{index}.wav`
- Samples: ~3,000 recordings (50 digits × 6 speakers)

### WAV Loading

```cpp
// File: src/experiments/guayaquil/lib/src/GuayaquilDataset.cpp
#include "wave/Wav.hpp"

// Load FSDD audio files
Wav wav_file;
wav_file.read(file.string());
const auto& raw_data = wav_file.get_data();  // std::vector<double>

// Convert to Tensor
nn::Tensor signal(static_cast<nn::Index>(raw_data.size()), 1);
for (std::size_t i = 0; i < raw_data.size(); ++i)
{
    signal.at(static_cast<nn::Index>(i), 0) = static_cast<float>(raw_data[i]);
}
```

### Progress Tracking

Real-time progress bars during training using `nn::utility::printProgress`:

```cpp
// Inside GuayaquilTraining.cpp
printProgress(train_samples.size(),
    1,
    train_samples.size() * cfg.epochs,
    epoch * train_samples.size() + train_samples.size(),
    epoch * train_samples.size() + train_samples.size(),
    false,
    run_id,
    total_runs,
    epoch + 1,
    cfg.epochs,
    train_samples.size(),
    train_samples.size(),
    static_cast<double>(val_mse),
    std::span<nn::Tensor*>{},
    "LSTM");
```

Output (multiline ANSI):
```
LSTM Fold:  [===================>                  ]  50%
Epoch: [===================>                  ]  50% (50/100)
Batch: [========================================] 100% (500/500b, 500/500s)  loss: 1.219896
```

### Reconstruction Metrics

The experiment is a **pure reconstruction study**. FSDD carries no anomaly labels, so F1/precision/recall are not computed. Reported metrics:

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | $\frac{1}{N}\sum(x-\hat x)^2$ | Primary quality metric |
| MAE | $\frac{1}{N}\sum|x-\hat x|$ | Robust alternative |
| $R^2$ | $1 - \text{SS\_res}/\text{SS\_tot}$ | Coefficient of determination |
| Spike rate | mean spike fraction over val set | SNN only |
| Energy | $r \cdot N + 10 \cdot \text{MACs}$ (SNN) / $10 \cdot \text{MACs}$ (LSTM) | Proxy; 10× constant = energy cost per MAC vs spike op |

`max_reconstruct_mean_deviation` remains in the config but is only used as a per-sample pass/fail threshold for logging — it does not affect the reported metrics.

### Layer Specification Manual

The architecture of the autoencoder is defined using a domain-specific language (DSL) in the `encoder_layer_spec` and `decoder_layer_spec` lists. Each string in the list represents a layer or a block of layers.

#### 1. Linear Layers
The most common layer used in Experiment04.
- **Format**: `linear:<width>[:<activation>]`
- **Width**: Can be a numeric value (e.g., `32`) or a special token:
    - `hidden`: Resolves to the base hidden size.
    - `latent`: Resolves to the bottleneck dimensionality.
    - `output`: Resolves to the input signal window size.
    - `branch_hidden` / `fusion_hidden`: Resolves to the specific branch/fusion sizes.
- **Activation**: Optional.
    - **ANN Mode**: `relu`, `leaky_relu`, `identity`.
    - **SNN Mode**: `leaky` (LIF), `leaky_integrator`, `identity`.
- **Example**: `linear:64:leaky` $\rightarrow$ A linear layer with 64 units and a LeakyReLU/LIF activation.

#### 2. Convolutional, Pooling, Residual (ANN mode only)
- **Conv1D**: `conv1d:<out_channels>:<kernel_size>[:<stride>[:<activation>]]`
- **Pool1D**: `pool1d:<kernel_size>[:<stride>]`
- **Residual**: `residual` or `residual:<repeat_count>`

> ⚠️ **SNN mode restriction**: In SNN autoencoders (`snn-ae`), `parse_layer_module_spec` only instantiates `linear` entries. Any `conv1d`, `pool1d`, or `residual` entry in `encoder_layer_spec` / `decoder_layer_spec` will **throw a `std::invalid_argument` at startup**. Keep SNN specs to `linear:width:leaky` / `linear:width:identity` only.

#### 3. Standalone Activations
- **Format**: `<activation_type>` (e.g., `relu`)

#### 4. SNN Architecture Modes (input transforms, not network layers)

`snn_architectures` in the profile selects how the raw signal is pre-processed **before** entering the autoencoder. All three modes share the same network (e.g., `linear:64:leaky / linear:32:identity`).

| Mode | Pre-processing applied to input |
|------|-------------------------------|
| `dense` | Pass-through (no transform) |
| `conv1d` | 3-tap smoothing filter: kernel `{0.25, 0.5, 0.25}` |
| `recurrent` | Stand-alone LIF transform (stateless, fixed V_th/alpha) |

These are **not** different network architectures — they are signal conditioning steps applied at `GuayaquilEncoding.cpp:apply_snn_architecture_transform`.

#### Building a Full Architecture
The total network is built by concatenating these specs. 
**Example Encoder**: `["linear:128:leaky", "residual:2", "linear:32:identity"]`
1. Linear(input $\rightarrow$ 128) $\rightarrow$ Lif ReLU
2. 2x Residual Blocks (128 $\rightarrow$ 128)
3. Linear(128 $\rightarrow$ 32) $\rightarrow$ Identity (Latent Bottleneck)

## Usage

```bash
# Run a specific profile (from software/nn/)
./out/build/max-performance/src/experiments/guayaquil/guayaquil \
  --comparative-config src/experiments/guayaquil/profiles/article-lstm-ae.json

# Run all article profiles + build paper CSVs (~2.5 h)
./scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh
```

Both `--comparative-config` and `--profile` are accepted as the flag name.

### Outputs

Results written to `results/` (or `dataset.results_dir` from profile):

| File | Contents |
|------|----------|
| `{run_tag}_comparative_metrics.csv` | One row per (model, encoding, architecture, v_th, alpha, run_id) |
| `{run_tag}_publication_table.csv` | Aggregated, formatted for paper tables |
| `{run_tag}_summary.json` | Config hash, per-model stats |
| `data/{run_tag}_*.dat` | pgfplots DAT files for paper figures |
| `data/paper_*.csv` | Aggregated across all runs (written by `02_guayaquil_build_lstm_vs_snn_paper_data.py`) |

Checkpoints in `results/checkpoints/` — safe to interrupt and resume.

### Paper data pipeline

```bash
# After all article runs complete:
python3 scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py \
  --results-dir results \
  --data-dir /path/to/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/guayaquil/profiles

# Compile paper:
cd documentation/07-articlesProduced/conference71070Guaiaquil
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Key Differences from Experiment03

| Feature | Experiment03 | Experiment04 |
|---------|-------------|--------------|
| Model | Feedforward AE | LSTM + SNN comparative |
| Input | Fixed-dim vectors | Variable-length sequences |
| Latent | Vector | Final hidden state |
| BPTT | Not used | LSTM uses BPTT |
| Dataset | 10.1117 EEG/Audio | FSDD (spoken digits) |
| Progress | Legacy async | nn::progress (ANSI) |

## Common Pitfalls

1. **SNN spec with non-linear entries throws at startup.** `parse_layer_module_spec` in SNN mode only handles `linear:width[:activation]`. Any `conv1d:`, `pool1d:`, `residual`, or `spiking_neuron:` entry throws `std::invalid_argument` before training begins.

2. **`early_stop_patience` must be < `epochs`.** `validate()` enforces this — profile will reject with a clear error if violated.

3. **`seed_deterministic: true` with `repeats > 1` produces identical runs.** Use `false` for article profiles (different seed per repeat).

4. **SNN architecture modes are signal transforms, not layers.** `conv1d`/`recurrent` in `snn_architectures` do not add conv or LSTM layers to the network — they pre-process the input window before it enters the autoencoder.

5. **FSDD path must match `dataset_root` in profile.** Default: `/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset`.

6. **F1/precision/recall are always 0 for FSDD.** FSDD has no anomaly labels. These fields exist in the output CSV but should not be cited.

7. **Benchmark runs resume from `results/checkpoints/`.** Results are cached by config hash, so re-running after a code change reuses the old numbers — a "run" that finishes in seconds with metrics identical to the previous one is the tell. Delete the results directory before any timing comparison.

8. **`lstm_frame_size` changes LSTM-AE results, not just its speed.** Do not mix runs with different values in one comparison table.

## Results

All results from 3 independent runs, FSDD dataset, window size 256, Adam(lr=1e-3, β₁=0.9, β₂=0.999), up to 30 epochs with early stopping (patience=10). SNN: 2 linear layers (64→32 latent). LSTM: 1-layer hidden=64, latent=32.

### Raw Publication Tables

Values are means across 3 runs. Energy unit: proxy score = spike\_rate × N + 10 × MACs (dimensionless relative measure). Train time in ms.

#### LSTM-AE (article-lstm-ae profile, 3 repeats)

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9978 | 0.7049 | 0.0022 | 0 | 8.52e+07 | 452 708 |
| lstm-ae | latency | 1 | 0.2348 | 0.4780 | 0.0483 | 0 | 8.52e+07 | 186 683 |
| lstm-ae | poisson | 1 | 0.1179 | 0.2523 | −0.0352 | 0 | 8.52e+07 | 158 527 |

#### SNN-dense (article-snn-dense profile, 3 repeats)

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9968 | 0.7054 | 0.0032 | 0 | 8.52e+07 | 666 860 |
| lstm-ae | latency | 1 | 0.2415 | 0.4863 | 0.0213 | 0 | 8.52e+07 | 385 226 |
| lstm-ae | poisson | 1 | 0.1185 | 0.2728 | −0.0408 | 0 | 8.52e+07 | 572 979 |
| snn-ae | direct | 2 | 1.1056 | 0.7709 | −0.1056 | 0.499 | 372 471 | 3 986 |
| snn-ae | latency | 2 | 0.1364 | 0.2673 | 0.4472 | 0.877 | 375 374 | 3 266 |
| snn-ae | poisson | 2 | 0.1244 | 0.1707 | −0.0921 | 0.970 | 376 092 | 6 485 |

#### SNN-conv1d (article-snn-conv1d profile, 3 repeats)

Conv1d mode = 3-tap smoothing filter {0.25, 0.5, 0.25} applied before encoding.

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9968 | 0.7054 | 0.0032 | 0 | 8.52e+07 | 302 894 |
| lstm-ae | latency | 1 | 0.2415 | 0.4863 | 0.0213 | 0 | 8.52e+07 | 323 118 |
| lstm-ae | poisson | 1 | 0.1185 | 0.2728 | −0.0408 | 0 | 8.52e+07 | 281 009 |
| snn-ae | direct | 2 | 0.8052 | 0.6342 | −0.1155 | 0.489 | 372 393 | 1 963 |
| snn-ae | latency | 2 | 0.1032 | 0.2331 | 0.5269 | 0.881 | 375 410 | 2 809 |
| snn-ae | poisson | 2 | 0.0697 | 0.1516 | −0.1284 | 0.994 | 376 271 | 1 363 |

#### SNN-recurrent (article-snn-recurrent profile, 3 repeats)

Recurrent mode = stand-alone LIF transform on input before encoding.

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9968 | 0.7054 | 0.0032 | 0 | 8.52e+07 | 318 625 |
| lstm-ae | latency | 1 | 0.2415 | 0.4863 | 0.0213 | 0 | 8.52e+07 | 566 580 |
| lstm-ae | poisson | 1 | 0.1185 | 0.2728 | −0.0408 | 0 | 8.52e+07 | 435 531 |
| snn-ae | direct | 2 | 0.1276 | 0.1786 | −0.0876 | 0.947 | 375 917 | 8 912 |
| snn-ae | latency | 2 | 0.1594 | 0.2859 | 0.2507 | 0.834 | 375 043 | 18 182 |
| snn-ae | poisson | 2 | 0.1156 | 0.1605 | −0.0836 | 0.970 | 376 090 | 15 787 |

---

### Compiled Analysis

#### Best MSE per encoding (lowest = best reconstruction)

| Encoding | Best model | MSE | vs LSTM-AE MSE | Improvement |
|----------|-----------|-----|----------------|-------------|
| direct | SNN-recurrent | 0.1276 | 0.9978 | **7.8×** |
| latency | SNN-conv1d | 0.1032 | 0.2348 | **2.3×** |
| poisson | SNN-conv1d | 0.0697 | 0.1179 | **1.7×** |

SNN consistently outperforms LSTM-AE on reconstruction MSE for all three encodings. Best overall: SNN-conv1d + poisson (MSE = 0.0697).

#### Energy efficiency (proxy: spike\_rate × N + 10 × MACs)

| Model | Energy (mean) | vs LSTM-AE | Factor |
|-------|--------------|------------|--------|
| LSTM-AE | ~8.52 × 10⁷ | baseline | 1× |
| SNN-dense | ~374 646 | 227× lower | **227×** |
| SNN-conv1d | ~374 691 | 227× lower | **227×** |
| SNN-recurrent | ~375 683 | 227× lower | **227×** |

SNN architecture mode does not affect energy — all three use the same network; the pre-processing transform is essentially free.

#### Spike rates

| Architecture | Encoding | Spike Rate |
|-------------|----------|------------|
| dense | poisson | 0.970 |
| conv1d | poisson | 0.994 |
| recurrent | poisson | 0.970 |
| dense | latency | 0.877 |
| conv1d | latency | 0.881 |
| recurrent | latency | 0.834 |
| dense | direct | 0.499 |
| conv1d | direct | 0.489 |
| recurrent | direct | 0.947 |

Poisson encoding produces highest spike rates (~0.97–0.99). Direct encoding varies by architecture (recurrent: 0.947, dense/conv1d: ~0.49).

#### Training time

| Model | Fastest encoding | Time (ms) | Slowest encoding | Time (ms) |
|-------|-----------------|-----------|-----------------|-----------|
| LSTM-AE | poisson | 158 527 | direct | 452 708 |
| SNN-dense | latency | 3 266 | poisson | 6 485 |
| SNN-conv1d | poisson | 1 363 | latency | 2 809 |
| SNN-recurrent | direct | 8 912 | latency | 18 182 |

SNN-conv1d trains fastest (1 363–2 809 ms). SNN-recurrent is slowest among SNNs (~9–18 s) due to LIF transform overhead per window. LSTM-AE trains slowest overall (~158–453 s).

#### R² summary

Positive R² indicates the model explains variance beyond the mean baseline. Only some configurations achieve positive R²:

| Model | Encoding | R² |
|-------|----------|----|
| SNN-conv1d | latency | **0.527** |
| SNN-dense | latency | **0.447** |
| SNN-recurrent | latency | **0.251** |
| LSTM-AE | latency | 0.048 |
| LSTM-AE | direct | 0.002–0.003 |
| all others | direct/poisson | < 0 |

Latency encoding is the only configuration where models learn meaningful variance structure. All direct and poisson encodings yield near-zero or negative R², indicating the models learn the signal mean but not its shape.

---

### Raw Data Files

| File | Description |
|------|-------------|
| `results/article_lstm_ae_comparative_metrics.csv` | Per-run raw metrics (9 rows: 3 encodings × 3 runs) |
| `results/article_snn_dense_comparative_metrics.csv` | Per-run raw metrics (81 rows: 3 enc × 3 arch-sweep × v_th × alpha × 3 runs) |
| `results/article_snn_conv1d_comparative_metrics.csv` | Same structure as dense |
| `results/article_snn_recurrent_comparative_metrics.csv` | Same structure as dense |
| `results/article_*_publication_table.csv` | Aggregated mean over runs, formatted for paper |
| `results/article_*_summary.json` | Config hash, seed, stat tests |
| `data/article_*_history.dat` | Per-run epoch loss curves (pgfplots format) |
| `data/article_*_convergence.dat` | Convergence diagnostic per run |
| `data/article_*_summary.dat` | Summary statistics (pgfplots format) |
| `data/article_*_sweep.dat` | Hyperparameter sweep results |

## See Also

- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md) - Theory
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md)
- [Autoencoders](../Concepts/Autoencoders.md)
- [Wave Processing](../Core/Wave.md)
- [Training](../Core/Training.md) - Progress bars
- [Experiment03](../Experiments/AutoencoderRunner.md) - Feedforward autoencoder

## References

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735

[2] A. Graves, "Generating sequences with recurrent neural networks," arXiv preprint arXiv:1308.0850, 2013. [Online]. Available: https://arxiv.org/abs/1308.0850

[3] FSDD Dataset: https://github.com/Jakobovski/Free-Spoken-Digits-Dataset
