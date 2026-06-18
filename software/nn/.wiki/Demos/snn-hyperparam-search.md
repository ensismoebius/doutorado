# SNN Hyperparameter Search

Suite of four Python hyperparameter search scripts for tuning a PyTorch + snnTorch SNN autoencoder. Implements a progressive staged strategy: coarse grid → staged refinement → targeted large-model search → k-fold cross-validation of top candidates.

---

## Theoretical Background

Staged hyperparameter search follows the successive halving (SHA) and HyperBand principles [Li et al., 2018]: allocate small budgets first, eliminate low performers, and concentrate compute on top candidates.

The Van Rossum distance [Rossum, 2001] is a biologically motivated spike train metric. Spike trains are convolved with an exponential kernel $\kappa(t) = e^{-t/\tau_c}$ before comparing with $L_2$ distance. Small $\tau_c$ = sensitive to precise spike timing; large $\tau_c$ = converges to rate-code MSE.

k-fold cross-validation [Kohavi, 1995] provides unbiased generalisation estimates, relevant for small imagined-speech corpora.

---

## How It Is Implemented Here

**Source:** `src/demos/pyDemos/snn_hyperparam_search/`

Four scripts with increasing search scope:

| Script | Combos | Epochs | Purpose |
|--------|--------|--------|---------|
| `run_hyper_search.py` | 24 | 3 | Quick elimination |
| `run_extensive_search.py` | 81 → top-5 | 8 → 80 | Two-stage |
| `run_targeted_search.py` | 36 | 80 | Large models |
| `run_with_validation.py` | 3 pre-selected | 10 | Val accuracy |
| `long_and_cv.py` | 1 best + top-3 | 30 + 10×5 | Long train + k-fold |

Search axes: `lr ∈ {1e-3, 5e-4, 1e-4}`, `hidden ∈ {32..1024}`, `loss_mode ∈ {rate, membrane, van_rossum, mse_vector}`, `num_passes ∈ {1..10}`.

```python
# SNN autoencoder (same across all scripts)
# Input x ∈ R^F → Poisson encode → spikes ∈ {0,1}^(T×F)
# Linear(F → hidden) → Leaky LIF (β learned)
# × num_passes
# Latent z ∈ R^hidden
# Linear(hidden → F) → Leaky LIF (β learned)
# Output spikes / membrane → loss_mode
```

---

## Data Flow

```mermaid
flowchart TD
    A["run_hyper_search.py\n 24 combinations, 3 epochs"] --> B["hp_results.csv"]
    B --> C["run_extensive_search.py\n Stage 1: 81 combos\n Stage 2: top-5, 80 epochs"]
    C --> D["extensive_stage1_results.csv\n extensive_stage2_results.csv"]
    D --> E["run_targeted_search.py\n Large models: hidden=256-1024"]
    E --> F["targeted_search.csv"]
    F --> G["run_with_validation.py\n Top-3 pre-selected, val_acc"]
    G --> H["long_and_cv.py\n 30 epochs + 5-fold CV"]
    H --> I["long_train_curve.csv\n cv_results.csv"]
```

---

## How to Build and Run

```bash
pip install torch snntorch numpy pandas scikit-learn

# Quick grid (24 combos, fast)
python src/demos/pyDemos/snn_hyperparam_search/run_hyper_search.py

# Extensive two-stage search
python src/demos/pyDemos/snn_hyperparam_search/run_extensive_search.py

# Targeted large-model search
python src/demos/pyDemos/snn_hyperparam_search/run_targeted_search.py

# Validation run (pre-selected configs)
python src/demos/pyDemos/snn_hyperparam_search/run_with_validation.py

# Long training + 5-fold CV
python src/demos/pyDemos/snn_hyperparam_search/long_and_cv.py
```

---

## Test Suite

No formal test binary. Validate by checking that `hp_results.csv` contains 24 rows and that `final_loss` decreases from epoch 1 to epoch 3 for most configurations.

---

## Common Pitfalls

1. **`van_rossum` loss with small $\tau_c$**: the Van Rossum kernel integrates over spike trains; very small $\tau_c$ makes the loss hyper-sensitive to exact spike timing, causing vanishing gradients when spikes are rare.
2. **Large hidden + many passes**: `run_targeted_search.py` explores `hidden=1024, num_passes=10`. This can exhaust GPU memory on a single 8 GB card. Reduce batch size or use `--device cpu` for these configurations.
3. **Non-reproducible results without seed**: the scripts use `seed=42` for train/val split but not always for weight initialisation. Re-runs may give slightly different final losses.

---

## See Also

- [Concepts/K-Fold-Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — k-fold theory
- [Concepts/SNN-and-Surrogate-Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — spiking network training
- [Concepts/Spike-Encoding](../Concepts/Spike-Encoding.md) — rate vs latency coding

---

## References

[1] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, "Hyperband: A novel bandit-based approach to hyperparameter optimization," *J. Mach. Learn. Res.*, vol. 18, no. 185, pp. 1–52, 2018.

[2] M. C. W. van Rossum, "A novel spike distance," *Neural Comput.*, vol. 13, no. 4, pp. 751–763, 2001.

[3] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in *Proc. IJCAI*, 1995, pp. 1137–1143.
