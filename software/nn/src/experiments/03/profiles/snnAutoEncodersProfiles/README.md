# SNN Autoencoder Hyperparameter Grid Search

## Overview
This directory contains 402 test profiles for systematic hyperparameter exploration of SNN autoencoders on the BaseDeDatosHablaImaginada dataset.

## Profiles Organization

### 1. Audio-Window-SNN (single modality - audio only)
- **Count**: ~260 profiles
- **Grid dimensions**:
  - Learning rates: [0.0001, 0.0003, 0.0005, 0.001] (4 values)
  - Batch sizes: [8, 16, 32, 64] (4 values)
  - Hidden sizes: [32, 64, 128] (3 values)
  - Latent sizes: [16, 32, 64] (3 values)
  - Loss functions: ["mse", "mae"] (2 values)
  - **Total combinations**: 4 × 4 × 3 × 3 × 2 = 288 profiles

**Rationale**: Audio-window is most stable (single modality). Use as primary baseline. MSE vs MAE loss critical for convergence comparison.

### 2. Fused-Window-SNN (multimodal - audio + EEG)
- **Count**: ~96 profiles
- **Grid dimensions**:
  - Learning rates: [0.0001, 0.0003, 0.0005] (3 values - more conservative)
  - Batch sizes: [4, 8, 16, 32] (4 values)
  - Hidden sizes: [64, 128] (2 values - reduced to focus)
  - Latent sizes: [32, 64] (2 values)
  - Loss functions: ["mse", "mae"] (2 values)
  - **Total combinations**: 3 × 4 × 2 × 2 × 2 = 96 profiles

**Rationale**: Multimodal adds complexity. Constrain learning rates and hidden/latent sizes. Explore if fusion benefits from larger capacity.

### 3. EEG-Window-SNN (single modality - EEG only)
- **Count**: ~18 profiles
- **Grid dimensions**:
  - Learning rates: [0.0001, 0.0003, 0.0005] (3 values)
  - Batch sizes: [8, 16, 32] (3 values)
  - Loss functions: ["mse", "mae"] (2 values)
  - Fixed: hidden_size=64, latent_size=32
  - **Total combinations**: 3 × 3 × 2 = 18 profiles

**Rationale**: EEG signals differ from audio (lower dimensionality). Use as validation that patterns generalize across modalities.

## Naming Convention
```
{modality}-snn_grid_{index:03d}_{lr}_{bs}_{hs}_{ls}_{loss}.json
```

Example: `audio-window-snn_grid_005_lr0.0003_bs16_hs64_ls32_mse.json`
- Modality: audio-window-snn
- Index: 005 (order created)
- Learning rate: 0.0003
- Batch size: 16
- Hidden size: 64
- Latent size: 32
- Loss: mse

## Key Hyperparameter Factors to Monitor

### 1. Learning Rate (LR)
- **Expected impact**: HIGH
- **Range**: 1e-4 to 1e-3
- **Hypothesis**: 
  - Too high (1e-3): Instability in SNN gradients, divergence
  - Too low (1e-4): Slow convergence, may not enter interesting regime
  - **Sweet spot expected**: 3e-4 to 5e-4 balance (from audio-window-snn baseline)
- **Analysis metric**: Epoch mean loss smooth downward trend

### 2. Batch Size (BS)
- **Expected impact**: MEDIUM
- **Range**: 4 to 64
- **Hypothesis**:
  - Smaller BS (4-8): Noisier gradients, may escape local minima, but slower
  - Larger BS (32-64): Faster per-epoch, smoother gradients, less exploration
  - SNN spike gradient estimates benefit from stable batches
- **Analysis metric**: Total wall-clock training time vs. final loss

### 3. Hidden Size (HS)
- **Expected impact**: MEDIUM-HIGH
- **Range**: 32 to 128
- **Hypothesis**:
  - Smaller (32): Under-parameterized bottleneck, weak reconstruction
  - Medium (64): Baseline, good TRX vs. overfitting balance
  - Larger (128): Risk of overfitting on small EEG/audio windows
- **Analysis metric**: Train/val loss gap

### 4. Latent Size (LS)
- **Expected impact**: MEDIUM
- **Range**: 16 to 64
- **Hypothesis**:
  - Smaller (16): High information compression, may lose structure
  - Medium (32): Baseline, often sufficient for temporal signals
  - Larger (64): More expressive, but may not improve reconstruction
- **Analysis metric**: Reconstruction quality per modality (val_eeg_loss, val_audio_loss)

### 5. Loss Function (Loss)
- **Expected impact**: HIGH
- **Options**: MSE vs MAE
- **Hypothesis**:
  - **MSE**: Standard, amplifies outliers. Can cause gradient explosion in SNNs with sparse spikes
  - **MAE**: Robust to outliers, sign-based gradients better match spike gradient heuristics
  - **Expected winner**: MAE for SNNs (reflects snntorch recommendation)
- **Analysis metric**: Loss variance, convergence speed

### 6. Modality (Modality)
- **Expected impact**: HIGH
- **Options**: Audio-only, EEG-only, Fused
- **Hypothesis**:
  - Audio (11025 features): Rich temporal info, good for SNN
  - EEG (256 features): Sparse but informative, lower quality signal
  - Fused: Multimodal fusion adds complexity, may need larger networks
- **Analysis metric**: Convergence speed, final loss

## Recommended Analysis Workflow

### Phase 1: Baseline Validation (Week 1)
1. Run audio-window-snn subset:  
   - LR ∈ [0.0001, 0.001]
   - BS ∈ [16, 32]  
   - HS = 64, LS = 32
   - Loss ∈ [MSE, MAE]
   - **Count**: 4 × 2 = 8 profiles
   - **Goal**: Confirm audio-window behavior, identify LR × Loss interaction

2. Identify best config (e.g., `lr0.0003_bs32_mse`)

### Phase 2: Modality Comparison (Week 2)
1. Apply best audio audio NN settings to EEG and Fused variants
2. Measure convergence characteristics per modality
3. Document scaling relationships (e.g., does fused need larger HS?)

### Phase 3: Deep Optimization (Week 3)
1. Run full audio-window-snn grid (all 288 profiles)
2. Create 2D heatmaps: Loss(LR, BatchSize), Loss(HS, LS), Loss(Loss, LR)
3. Identify parameter interactions and local optima

### Phase 4: Generalization (Week 4)
1. Apply top-5 audio-window configs to fused-window and eeg-window
2. Measure generalization gap
3. Determine modality-specific tweaks needed

## Execution Notes

### Environment
- Dataset: BaseDeDatosHablaImaginada (EEG + Audio)
- Epochs per profile: 8 (quick iteration)
- Max batches per epoch: 30 (subset for speed)
- K-fold: Disabled for faster iteration

### Cluster / Batch Submission
To run all profiles, use a job scheduler:
```bash
#!/bin/bash
for profile in src/experiments/03/profiles/tests/*.json; do
    timeout 300 ./out/build/.../experiment03 --profile "$profile"
    sleep 5  # Cooldown
done
```

Or parallelize with GNU Parallel:
```bash
ls src/experiments/03/profiles/tests/*.json | \
  parallel -j 4 "timeout 300 ./out/build/.../experiment03 --profile {}"
```

### Results Aggregation
After each batch, run analysis:
```python
import json
import os
import pandas as pd

results = []
for f in os.listdir("src/experiments/03/results"):
    if f.endswith(".json"):
        with open(f"src/experiments/03/results/{f}") as fp:
            data = json.load(fp)
            results.append({
                "profile": data["profile"],
                "exit_code": data["exit_code"],
                "final_train_loss": data["epoch_mean_losses"][-1] if data["epoch_mean_losses"] else None,
                "final_val_loss": data.get("mean_val_loss", None),
                "batches": data["seen_batches"],
            })

df = pd.DataFrame(results).sort_values("final_val_loss")
print(df.head(20))
df.to_csv("results_summary.csv")
```

## Expected Findings

### Hypothesis 1: MSE vs MAE
**Prediction**: MAE converges faster and more stably due to sign-based gradient compatibility with SNN surrogate gradients.
**Success metric**: MAE final loss < MSE final loss, MAE loss variance lower

### Hypothesis 2: Learning Rate for SNNs
**Prediction**: SNNs more sensitive to LR than standard ANNs due to gradient sparsity.
**Success metric**: Optimal LR narrower range (3e-4 to 5e-4), not 1e-4 to 1e-3

### Hypothesis 3: Multimodal Fusion
**Prediction**: Fused-window-snn needs larger hidden size (128) vs audio-only (64) to combine modalities effectively.
**Success metric**: Fused HS=128 outperforms HS=64, but cost is higher than audio-only

### Hypothesis 4: Modality Informativeness
**Prediction**: Audio > EEG in terms of reconstruction quality and convergence speed.
**Success metric**: Audio final loss ~20-30% lower than EEG, fused intermediate

## Output Format

Each profile execution creates a JSON result in `src/experiments/03/results/` with:
```json
{
    "profile": "audio-window-snn_grid_005_...",
    "dataset_type": "audio-window",
    "autoencoder_type": "audio-window-snn",
    "exit_code": 0,
    "epoch_mean_losses": [0.72, 0.71, 0.70, ...],
    "mean_val_loss": 0.69,
    "final_learning_rate": 0.00015,
    "processed_samples": 12400
}
```

### Quick Aggregation Command
```bash
jq -s 'map({profile, exit_code, final_loss: .epoch_mean_losses[-1]}) | sort_by(.final_loss) | .[0:10]' \
  src/experiments/03/results/2026*.json
```

## Next Steps

1. **Execute**Phase 1-2 profiles
2. **Visualize** heatmaps of Loss(LR, BS) and Loss(Loss, Modality)
3. **Document** empirical findings vs. hypotheses
4. **Identify** top-3 configs per modality
5. **Deploy** best config to production training runs

---
**Generated**: 2026-04-12  
**Total profiles**: 402  
**Modalities**: Audio-window (260), Fused-window (96), EEG-window (18), Protocol (pending fix)
