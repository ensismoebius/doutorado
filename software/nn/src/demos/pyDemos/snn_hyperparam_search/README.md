# SNN Hyperparameter Search (snn_hyperparam_search)

Suite of four hyperparameter search scripts for tuning a PyTorch + snnTorch SNN autoencoder. Implements a progressive staged strategy: coarse grid → staged refinement → targeted large-model search → k-fold cross-validation of top candidates. Targets the optimal combination of learning rate, hidden size, spike loss mode, and number of encoding passes.

## Algorithm

### SNN autoencoder training objective

All scripts train the same SNN autoencoder. Input $\mathbf{x} \in \mathbb{R}^F$ is Poisson-encoded into spike trains over $T$ time steps and reconstructed. Loss mode controls the objective:

**rate**: spike count loss — minimise $\mathcal{L} = \|\bar{S}_\text{out} - \mathbf{x}\|^2$ where $\bar{S}_\text{out} = \frac{1}{T}\sum_t S_\text{out}[t]$.

**membrane**: membrane potential MSE — $\mathcal{L} = \|V_\text{out} - \mathbf{x}\|^2$.

**van_rossum**: Van Rossum distance (Rossum, 2001) — convolves spike trains with exponential kernel $\kappa(t) = e^{-t/\tau_c}$ before computing $L_2$ distance:

$$\mathcal{L}_\text{vR} = \frac{1}{2\tau_c} \int_0^T \!\!\!\left(\sum_{t_i} \kappa(t - t_i^{(a)}) - \sum_{t_j} \kappa(t - t_j^{(b)})\right)^2 \!\!\! dt$$

**mse_vector**: direct MSE between output spike vector and target vector.

### Poisson encoding

Input normalised to $[0, 1]$; spike probability at each step $= x_i$. Rate coding: $S_i[t] \sim \text{Bernoulli}(x_i)$ over $T$ steps.

## Scripts and Search Spaces

### `run_hyper_search.py` — Quick Grid

```
lr         ∈ {1e-3, 5e-4}
hidden     ∈ {32, 64}
loss_mode  ∈ {rate, membrane, van_rossum}
num_passes ∈ {1, 2}
─────────────────────────────
Total: 24 combinations
Epochs: 3 | Samples: 200 subsample
Output: hp_results.csv
```

Fastest scan for eliminating bad configurations. Writes all 24 rows with final loss.

### `run_extensive_search.py` — Two-Stage

**Stage 1:**
```
lr         ∈ {1e-3, 5e-4, 1e-4}
hidden     ∈ {32, 64, 128}
loss_mode  ∈ {rate, membrane, van_rossum}
num_passes ∈ {1, 2, 3}
─────────────────────────────
Total: 81 combinations | Epochs: 8 | Samples: 800
```

**Stage 2:** top-5 configs by final loss, retrained on full dataset:
```
Epochs: 80 | Early stop: loss < 0.01
```

### `run_targeted_search.py` — Large Models

```
lr         ∈ {1e-3, 5e-4, 1e-4}
hidden     ∈ {256, 512, 1024}
loss_mode  ∈ {mse_vector, van_rossum}
num_passes ∈ {5, 10}
─────────────────────────────
Total: 36 combinations
Epochs: 80 | Early stop: loss < 0.01
Output: targeted_search.csv
```

Targets the hypothesis that wider networks with more encoding passes reduce reconstruction loss.

### `run_with_validation.py` — Top-3 Validation

Pre-selected best configs from prior search:

| lr | hidden | loss_mode | num_passes |
|---|---|---|---|
| 1e-3 | 64 | van_rossum | 2 |
| 5e-4 | 64 | van_rossum | 2 |
| 1e-3 | 32 | membrane | 1 |

Train/val split: 80 / 20 (seed=42), 10 epochs. Evaluates `val_acc` via Poisson-encoded spike inference + spike count classification.

### `long_and_cv.py` — Long Training + K-fold CV

**Long training:** best config `{lr=1e-3, hidden=32, loss_mode=van_rossum, num_passes=1}`, 30 epochs, 80/20 split. Writes `long_train_curve.csv`.

**5-fold CV:** top-3 candidates, 10 epochs per fold. Writes `cv_results.csv` with per-fold train loss and val accuracy.

## Architecture

SNN autoencoder (same across all scripts):

```
Input x ∈ R^F
    │
Poisson encode  →  spikes ∈ {0,1}^{T × F}
    │
Linear(F → hidden)  →  Leaky LIF (β learned)
    ×  num_passes
    │
Latent z ∈ R^hidden
    │
Linear(hidden → F)  →  Leaky LIF (β learned)
    │
Output spikes / membrane  →  loss_mode
```

`β = exp(-dt/(R·C))` initialised from config; learnable via snnTorch's `Leaky(beta=β, learn_beta=True)`.

## Theory & State of the Art

The Van Rossum distance (Rossum, 2001) is a biologically motivated spike train metric that extends rate-coding loss to account for spike timing. Unlike raw spike count MSE, it penalises temporal jitter via the exponential kernel width $\tau_c$: small $\tau_c$ makes it sensitive to precise spike timing (temporal coding limit); large $\tau_c$ converges to rate-code MSE.

Staged hyperparameter search follows the successive halving (SHA) and HyperBand principles (Li et al., 2018): allocate small budgets first, eliminate low performers, and concentrate compute on top candidates. This approach reduces total wall-clock time by a factor of $O(\log(\eta \cdot n))$ relative to full grid search.

For SNNs, learning rate and hidden size interact strongly with the spike loss: membrane-potential MSE provides dense gradients but ignores temporal structure; rate loss is sparse and biologically meaningful but suffers from the dead-neuron problem; Van Rossum loss strikes a balance. The targeted search focuses on `mse_vector` and `van_rossum` based on preliminary results in Stage 1.

k-fold cross-validation (Kohavi, 1995) provides unbiased generalisation estimates for small datasets, which is relevant for imagined-speech EEG corpora with typically $<$50 subjects.

## How to Use (HOWTO)

### Requirements

```bash
pip install torch snntorch numpy pandas scikit-learn
```

### Quick grid (24 combos, fast)

```bash
python src/demos/pyDemos/snn_hyperparam_search/run_hyper_search.py
```

Output: `hp_results.csv` in working directory.

### Extensive two-stage search

```bash
python src/demos/pyDemos/snn_hyperparam_search/run_extensive_search.py
```

Output: `extensive_stage1_results.csv`, `extensive_stage2_results.csv`.

### Targeted large-model search

```bash
python src/demos/pyDemos/snn_hyperparam_search/run_targeted_search.py
```

Output: `targeted_search.csv`.

### Validation run

```bash
python src/demos/pyDemos/snn_hyperparam_search/run_with_validation.py
```

Output: console table with `val_acc` per config.

### Long training + 5-fold CV

```bash
python src/demos/pyDemos/snn_hyperparam_search/long_and_cv.py
```

Output: `long_train_curve.csv`, `cv_results.csv`.

### Expected CSV columns

`hp_results.csv` / `targeted_search.csv`:

```
lr, hidden, loss_mode, num_passes, epoch, train_loss, final_loss
```

`cv_results.csv`:

```
config_id, fold, train_loss, val_accuracy
```

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Neural network layers, Adam, BPTT |
| `snntorch` | `Leaky` spiking neurons, Poisson encoding, Van Rossum loss |
| `numpy` | Array operations |
| `pandas` | CSV result logging |
| `scikit-learn` | `KFold`, train/test split |
