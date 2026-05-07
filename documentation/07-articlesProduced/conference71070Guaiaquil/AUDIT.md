# AUDIT: Paper × Profiles × Code Three-Way Diff

Date: 2026-05-07. Generated for ETCM 2026 submission readiness check.

Legend: ✓ matches · ✗ broken (functional bug) · ⚠ drift (doc only / fixable text)

| # | Topic | Paper says | Profile says | Code does | Status | Owner-fix |
|---|---|---|---|---|---|---|

## §1 Dataset (FSDD)

|  1 | Dataset name | FSDD | `fsdd` | reads `*.wav` from `dataset_root` | ✓ | — |
|  2 | Sample rate | 8 kHz (text) | not configured | not enforced; uses raw PCM | ⚠ | paper |
|  3 | Window size T | 256 (32 ms @ 8 kHz) | 256 | `cfg.dataset.window_size` | ✓ | — |
|  4 | Train windows cap | "up to 100" | 100 | `max_loaded_train_samples + max_validation_samples` total cap | ✗ | code (C2) |
|  5 | Val windows cap | "up to 30" | 30 | `min(max_val, total/5)` — silently truncated | ✗ | code (C2) |
|  6 | Normalization | "per-sample (x_min, x_max)" | not configured | `zscore_inplace` per window (mean/std, NOT min/max) | ✗ | paper (H2) |
|  7 | Windowing stride | "non-overlapping, S = T" | not configured | `offset += window_size` ✓ | ✓ | — |
|  8 | Pool + shuffle + split | "all windows pooled, shuffled, split" | not configured | files sorted, windows accumulated in file order, **no shuffle**; val = last `total/5` | ✗ | code (C6) |
|  9 | Validation labels | "F1 / precision / recall reported" | not configured | **synthetic**: every 10th val sample → label=1 + `+1.5` offset | ✗ | paper (C1) |

## §2 Encodings

| 10 | Direct | $\tilde x = x$ | `direct` | passthrough | ✓ | — |
| 11 | Poisson | Bernoulli on clipped $x/x_{max}$ | `poisson` | mt19937 seeded per-sample, Bernoulli on `[0, max]` | ✓ | — |
| 12 | Latency | TTFS in $[1, T-1]$ | `latency` | `(1-scaled)*(T-1)` clamped `[1, T-1]` | ✓ | — |
| 13 | Encoding seed | "fixed seeds 42,43,44" | `seed: 42, repeats: 3, seed_deterministic: false` | `seed + run_id`; seed_deterministic ignored when false → `seed + run_id*1000`? verify | ⚠ | code/profile audit |

## §3 LSTM-AE Architecture

| 14 | Stacked LSTM layers | "two LSTM layers (H=64)" | `encoder_layer_spec: [lstm:64, linear:64:leaky, linear:32:tanh]` → 2 linear entries → `num_layers=2` | `LSTMAutoencoder` builds `num_layers=2` LSTMLayers ✓ | ✓ | — |
| 15 | Hidden size H | 64 | `lstm_hidden_size: 64` | `arch.hidden_size = 64` | ✓ | — |
| 16 | Latent dim d | 32 | `latent_dim: 32` | `arch.latent_size = 32` | ✓ | — |
| 17 | Latent activation | "Linear(H→d)→tanh" | `linear:32:tanh` | hardcoded `tanh` after `Linear(H,Z)` projection | ✓ | — |
| 18 | Forget bias init | (paper: 1.0 [3]) | `forget_bias_init: 1.0` (dead) | `LSTMLayer` ctor sets `b_[H:2H]=1.0` ✓ | ✓ | — |
| 19 | Weight init | not specified | not configured | `N(0, 0.05)`, `mt19937(42 + offset)` | ⚠ | paper (add line) |
| 20 | Decoder | "Linear(d→H) replicate T times → 2 stacked LSTMs → Linear(H→T)" | `[linear:64:leaky, lstm:64, linear:output:identity]` | matches | ✓ | — |

## §4 SNN-AE Architecture

| 21 | Layer count | "two-layer fully connected" | dense: `[linear:64, spiking_neuron:leaky:64, linear:32]` | builder THROWS on `spiking_neuron:leaky:64` (3 tokens, not in parser switch) | ✗ | profile (C7) |
| 22 | Neuron model | LIF (`Leaky`) + readout `LeakyIntegrator` | `model_type: snn_autoencoder_*` (dead) | `Leaky` for hidden; `LeakyIntegrator` for output if `activation == "identity"` | ✓ | — |
| 23 | Surrogate | exponential SuperSpike, $\sigma_s=1$ | `surrogate_gradient_function: exponential` (dead) | `Leaky.hpp` includes only `ExponentialSurrogate` ✓ | ✓ | — |
| 24 | LIF decay $\beta$ | $\beta = e^{-\Delta t / RC}$ | not configured | `time_step=1`, `R = 1/v_th`, `C = -1/ln(α)` → β=α ✓ | ✓ | — |
| 25 | R, C clamping | "$R, C$ clamped to $10^{-6}$" | not configured | clamp in forward ✓ | ✓ | — |
| 26 | Reset | hard or soft reset | `reset_mechanism: "hard_reset (V→0)"` (dead) | hard reset (verify in Leaky.hpp) | ⚠ | confirm |
| 27 | $V_{th}$ values | {0.5, 1.0, 1.5} | `v_th_values: [0.5, 1.0, 1.5]` | iterated in outer loop ✓ | ✓ | — |
| 28 | $\alpha$ values | {0.8, 0.9, 0.99} | `alpha_values: [0.8, 0.9, 0.99]` | iterated ✓ | ✓ | — |

## §5 SNN Sub-architectures (dense / conv1d / recurrent)

| 29 | "Dense" mode | "$\phi(\tilde x)=\tilde x$ (identity pre-process)" | `snn_architectures: [dense]` | `apply_snn_architecture_transform` no-op for unknown/`dense` ✓ | ✓ | — |
| 30 | "Conv1d" mode | "1D-convolutional pre-processor with spiking neurons" | `encoder_layer_spec: [conv1d:64:kernel=3:stride=1:padding=1, ...]` | `conv1d_temporal_smooth`: hardcoded 3-tap `{0.25, 0.5, 0.25}` filter; **NOT a Conv1D layer** | ✗ | paper (C3), profile |
| 31 | "Recurrent" mode | "LSNN (recurrent SNN) with adaptive threshold" | `encoder_layer_spec: [recurrent:lsnn:64, linear:32]`, `threshold_adaptation: true` | `recurrent_lif_encode`: stand-alone LIF integration on input; no adaptive threshold; not an LSNN layer | ✗ | paper (C4), profile |
| 32 | Conv1d kernel size | 3 (paper) | `kernel=3:stride=1:padding=1` | hardcoded 3-tap, no kernel parameter | ⚠ | paper (clarify) |
| 33 | LSNN adaptive a[t] | "a[t] = d·a[t-1] + c·s[t-1]" | `adaptation_param_a: "learnable d, c"` (dead) | not implemented | ✗ | paper (C4) |
| 34 | SNN layer_spec parser | (paper does not describe spec language) | `[conv1d:64:kernel=3:stride=1:padding=1, ...]`, `[recurrent:lsnn:64, ...]`, `[spiking_neuron:leaky:64]` | parser supports `linear`, `conv1d` (4-token form), `pool1d/2d`, `residual`, `lstm`, single-token activation. `spiking_neuron:*:*` (3 tokens) = THROW. `recurrent:lsnn:64` (3 tokens, head not matched) = THROW. `kernel=3:stride=1:padding=1` (key=value) = malformed. | ✗ | profile (C7) |

## §6 Training

| 35 | Optimizer | Adam | `optimizer: adam` (dead) | `Trainer` builds Adam from TrainerConfig | ✓ | — |
| 36 | β₁, β₂ | 0.9, 0.999 | `beta1: 0.9, beta2: 0.999` (also `adam_beta1`/`adam_beta2`) | parser reads both keys; default 0.9/0.999 ✓ | ✓ | — |
| 37 | ε | $10^{-8}$ | `epsilon: 1e-8` | parser reads, default 1e-8 ✓ | ✓ | — |
| 38 | Learning rate η | $10^{-3}$ | `learning_rate: 0.001` | TrainerConfig.learning_rate ✓ | ✓ | — |
| 39 | Bio-param lr | $0.1\eta = 10^{-4}$ | `learning_rate_biophysical: 0.0001` | `snn_lr_scale = lr_bio / lr` for SNN; LSTM uses 1.0 ✓ | ✓ | — |
| 40 | Batch size | 1 | `samples_per_batch: 1` | TrainerConfig.batch_size; LSTM forces 1 in code ✓ | ✓ | — |
| 41 | Max epochs | 30 | `epochs: 30` | TrainerConfig.epochs ✓ | ✓ | — |
| 42 | Early-stop patience | 5 | `early_stop_patience: 5` | passed to `EarlyStoppingCallback(patience)` ✓ | ✓ | — |
| 43 | Early-stop min_delta | $10^{-8}$ | not configured | hardcoded `1e-8F` in `EarlyStoppingCallback.hpp:15` | ✓ (matches by coincidence) | code optionally (H3) |
| 44 | Loss | MSE reconstruction | `loss: mse`, `model.loss_function: mse` | Trainer hardcoded `MSELossImpl<Backend>`; `cfg.model.loss_type` is dead config | ✓ (effective behaviour) ✗ (config dead) | code wire-up or profile cleanup (H1) |
| 45 | Surrogate scale | $\sigma_s=1$ | `surrogate_gradient_scale: 1.0` (dead) | Exponential surrogate; verify scale parameter | ⚠ | confirm |

## §7 Sweeps

| 46 | Repeats / seeds | 3 seeds {42, 43, 44} | `repeats: 3, seed: 42, seed_deterministic: false` | seeds 42+run_id; LSTM 9 = 3×3; SNN 81 = 3×3×3×3 ✓ | ✓ | — |
| 47 | LSTM total runs | 9 | `_total_runs_breakdown: 9` | matches | ✓ | — |
| 48 | SNN dense total runs | 81 | `_total_runs_breakdown: 81` | matches | ✓ | — |
| 49 | SNN conv1d total runs | 81 | 81 | matches | ✓ | — |
| 50 | SNN recurrent total runs | 81 | 81 | matches | ✓ | — |
| 51 | Backend-bench | (not described) | `seed_deterministic: true` (drift), nested `model.lstm_model`/`model.snn_model` | parser is flat → nested fields ignored, defaults used | ✗ | profile (H5, H6) |

## §8 Metrics

| 52 | MSE / MAE / R² | computed | not configured | `mse_between`, `mae_between`, R² in `evaluate_*` ✓ | ✓ | — |
| 53 | Precision/Recall/F1 | computed (text) | not configured | computed against **synthetic** val_labels (see #9) | ✗ | paper (C1) |
| 54 | F1 threshold $\delta$ | 0.25 | `max_reconstruct_mean_deviation: 0.25` | `evaluate_*` uses cfg ✓ | ✓ | — |
| 55 | Spike rate | "$r$ = fraction of spikes" | not configured | `spike_sum / n_values` over output (LSTM=0; SNN=mean firing rate) ✓ | ✓ | — |
| 56 | Energy formula (LSTM) | "energy ∝ MAC count" (Section IV) | not configured | `energy = 10 * macs` (constant 10 not in paper) | ⚠ | paper (state constant) |
| 57 | Energy formula (SNN) | "energy ∝ spike rate r and MAC count M" | not configured | `energy = spike_rate * n_values + 10 * macs` (additive, not product; constants not in paper) | ⚠ | paper (state formula) |
| 58 | LSTM MACs | not specified | not configured | `T*(4*H*(I+H)*L) + (H*Z + Z*H + H*I)` | ⚠ | paper (add formula or omit) |
| 59 | SNN MACs | not specified | not configured | `F*H + (L-1)*H² + H*F` | ⚠ | paper (add formula or omit) |
| 60 | Param count | counted | not configured | `parameter_count(model.params())` ✓ | ✓ | — |

## §9 Backend

| 61 | XTensor preset | "xtensor + CBLAS, SIMD" | `backend: xtensor` (dead, derived at compile) | `active_backend_name()` returns "xtensor" by default ✓ | ✓ | — |
| 62 | OpenCL preset | "AMD/rusticl, lazy sync" | (separate preset) | conditional compile — not run yet for article | ⚠ | scope |

## §10 Output files (paper expects vs code writes)

| 63 | Profiles table | `data/paper_profiles.csv` | — | code writes `{run_tag}_profile_manifest.csv` (one per profile) | ✗ | aggregator (H4) |
| 64 | Summary by model | `data/paper_summary_by_model.csv` | — | code writes `{run_tag}_summary_by_model.csv` | ✗ | aggregator |
| 65 | Recon by encoding | `data/paper_recon_by_encoding.csv` | — | code writes `{run_tag}_recon_by_encoding.csv` ✓ schema | ✗ | aggregator |
| 66 | Eff by encoding | `data/paper_eff_by_encoding.csv` | — | code writes `{run_tag}_efficiency_by_encoding.csv` (note: `efficiency` not `eff`) | ✗ | aggregator (rename in paper or code) |
| 67 | Backend comparison | `data/paper_backend_comparison.csv` (cols `operation, xtensor_ms, opencl_ms, speedup`) | — | code writes `{run_tag}_backend_timing.csv` (cols `backend, model, train_ms, infer_ms`) — schema differs | ✗ | aggregator + script |
| 68 | Sweep alpha | `data/paper_sweep_alpha.csv` | — | code writes `{run_tag}_sweep_alpha.csv` | ✗ | aggregator |
| 69 | MSE plot | `data/paper_mse_plot.csv` | — | code writes `{run_tag}_mse_plot.csv` | ✗ | aggregator |
| 70 | Per-run history DAT | not referenced | `latex_data_dir` set | code writes `{run_tag}_*_history.dat`, `*_convergence.dat` | ⚠ | confirm not orphaned |

## §11 Profile dead-config drift (M1, profile-only)

The following profile keys are stored but **never reach `ComparativeConfig` fields**:

| Section | Keys |
|---|---|
| `training` | `loss`, `loss_reduction`, `optimizer`, `snn_param_scaling`, `surrogate_gradient_scale`, `adaptive_threshold_enabled` |
| `model` | `model_type`, `loss_function` (parsed but unused downstream — see #44), `loss_reduction`, `surrogate_gradient_function`, `surrogate_gradient_width`, `membrane_decay_beta` (string), `threshold_v_th` (string), `threshold_adaptation`, `reset_mechanism`, `biophysical_params_learnable`, `biophysical_params_lr_scale`, `adaptation_param_a`, `training_algorithm`, `forget_bias_init`, `encoder_description`, `decoder_description` |

These mislead reviewers reading the profile.

---

## Summary by severity

| Severity | Count | Findings |
|---|---|---|
| ✗ Functional | 11 | rows 4, 5, 6, 8, 9, 21, 30, 31, 33, 34, 51, 63–69 |
| ⚠ Drift / doc | 16 | rows 2, 13, 19, 26, 32, 45, 56–59, 61, 62, 70, plus M1 keys |
| ✓ Matches | 33 | the rest |

## Recommended action plan (deadline 2026-05-10)

| Priority | Action | Effort |
|---|---|---|
| P0 | C1: drop F1/precision/recall from paper Section V (synthetic labels). Mention in Limitations as "pure-reconstruction study". | 30 min paper text |
| P0 | C3: rename "1D-convolutional pre-processor" → "3-tap temporal smoothing pre-filter" in paper §III; update profile `_*` doc strings | 20 min |
| P0 | C4: rename "LSNN encoder with adaptive threshold" → "stand-alone LIF input transform" in paper §III; remove `threshold_adaptation: true` from profile | 20 min |
| P0 | C7: fix all 4 SNN profile `encoder_layer_spec` / `decoder_layer_spec` to valid format that the builder accepts | 15 min |
| P0 | C2 + C6: fix `ComparativeDataset.cpp` — use `cfg.dataset.max_validation_samples` directly; shuffle pool before split | 20 min code + rebuild |
| P1 | H2: paper text on normalisation ("z-score per window before encoding") | 5 min |
| P1 | H4: aggregator script `aggregate_paper_csvs.py` reading all `{run_tag}_*.csv` from `latex_data_dir` and writing `paper_*.csv` | 60 min |
| P1 | H6: set `seed_deterministic: false` in backend-bench profile (consistency only) | 1 min |
| P2 | M1: strip dead-config description fields from profiles | 15 min |
| P2 | New `config_round_trip_gtest.cpp` to prevent silent drift | 45 min |
| P2 | Paper §IV: add precise energy + MAC formulas (matches code) | 15 min paper text |

**Total estimated effort: ~4 hours.**

## Out of scope (explicit)

- Implementing real Conv1D layer + LSNN with adaptive threshold (C3/C4 functional fix): >2 days.
- Real-data anomaly labels for F1 (C1 functional fix): FSDD has none.
- OpenCL backend run.
