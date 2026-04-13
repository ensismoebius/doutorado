# SNN Grid Search Re-Run Analysis Plan

## Execution Details

**Current Run:** 20260413_000614  
**Status:** Running (started ~00:06:14)  
**Configuration:**
- Total profiles: 402 (288 audio-window, 96 fused-window, 18 eeg-window)
- Concurrent jobs: 12
- Timeout per profile: 600 seconds (10 minutes) — **Increased from 180s to allow convergence**
- Log level: warn (reduced I/O overhead vs. info)
- Thread limits: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1

**Expected Duration:**
- Average per profile: ~240-310 seconds (observed from previous partial run)
- Total estimated: 16-22 hours unattended

**Tracking Files:**
- Job log: `logs/grid_runs/20260413_000614_joblog.tsv` (updated live)
- Results directory: `logs/grid_runs/20260413_000614_results/`
- Completion marker: `logs/grid_runs/20260413_000614_DONE`

## Analysis Tables (Generated Upon Completion)

Once all 402 profiles complete, run this command to generate comprehensive CSV tables:

```bash
python3 scripts/analyze_grid_results.py
```

This creates the following output files in `analysis/` directory:

### 1. **master_comparison.csv** — Complete Ranking
- **Columns:** Rank, Profile, Modality, Loss Type, LR, BS, HS, LS, Final Train Loss, Final Val Loss, Mean Val Loss, Best Val Loss, Epochs, Status, Error
- **Purpose:** Full leaderboard of all 402 profiles ranked by mean validation loss
- **Use case:** Identify top performers across all modalities and loss types

### 2. **comparison_{modality}.csv** — Per-Modality Rankings
- **Files:**
  - `comparison_audio-window.csv` (288 profiles)
  - `comparison_fused-window.csv` (96 profiles)
  - `comparison_eeg-window.csv` (18 profiles)
  - `comparison_protocol.csv` (if applicable)
- **Columns:** Rank, Loss Type, LR, BS, HS, LS, Final Train Loss, Mean Val Loss, Best Val Loss, Epochs
- **Purpose:** Modality-specific performance analysis

### 3. **sensitivity_{modality}_{loss}.csv** — Hyperparameter Analysis
- **Files:** One per unique (modality, loss_type) combination
- **Columns:** Hyperparameter, Value, Avg Val Loss, Best Val Loss, Count
- **Purpose:** Understand impact of each hyperparameter setting
- **Analysis:** Shows average loss for each LR, BS, HS, LS value across all other configurations

### 4. **summary_statistics.csv** — Key Metrics
- **Contents:**
  - Total profiles, success/failure counts, success rate
  - Global statistics (best/worst/average/median loss)
  - Per-modality best performance and sample counts
- **Purpose:** High-level project summary and health metrics

## Expected Results (Based on Previous Partial Run)

**Early Findings from 46/402 profiles:**
- **Audio-Window MAE:** Mean val loss ~0.779 (excellent)
- **Audio-Window MSE:** Mean val loss ~0.997 (baseline, higher as expected)
- **Convergence:** All profiles reach 8 epochs within ~300 seconds
- **Exit Rate:** No timeouts observed with generous time allocation

**Predicted Patterns:**
- MAE loss consistently outperforms MSE by ~0.22 loss units on same architecture
- Lower learning rates (0.0001) likely more stable than higher values
- Batch size effects will be visible in sensitivity table
- Hidden/latent sizes show diminishing returns above 32/16

## Data Dictionary

### Hyperparameter Ranges
- **Learning Rate (LR):** 0.0001, 0.0005, 0.001
- **Batch Size (BS):** 4, 8, 16, 32, 64
- **Hidden Size (HS):** 32, 64, 128
- **Latent Size (LS):** 16, 32, 64

### Modalities
- **audio-window:** Input from sliding audio window
- **fused-window:** Audio + EEG fusion
- **eeg-window:** EEG-only input
- **protocol:** Full dataset protocol (if included)

### Loss Types
- **mae:** Mean Absolute Error
- **mse:** Mean Squared Error

### Key Metrics
- **Final Train Loss:** Training loss at last epoch
- **Final Val Loss:** Validation loss at last epoch
- **Mean Val Loss:** Average validation loss across all K-fold splits (primary ranking criterion)
- **Best Val Loss:** Minimum validation loss observed during training
- **Epochs:** Number of training epochs completed

## Monitoring Progress

Check progress while batch runs:

```bash
# Current job count
ps aux | grep experiment03 | wc -l

# Live job log (updates every ~60s)
tail -20 logs/grid_runs/20260413_000614_joblog.tsv

# Result file count
ls src/experiments/03/results/*grid* | wc -l

# Check for completion marker
cat logs/grid_runs/20260413_000614_DONE 2>/dev/null && echo "Done!" || echo "Still running..."
```

## Next Steps After Batch Completes

1. **Run analysis:** `python3 scripts/analyze_grid_results.py`
2. **Review master table:** Open `analysis/master_comparison.csv` to see full rankings
3. **Deep dive:** Examine specific modality or sensitivity tables as needed
4. **Archive results:** Entire `analysis/` directory is ready for version control and documentation

## Notes

- All results are JSON-formatted in `src/experiments/03/results/` with full metadata
- Each CSV table is independent; can be analyzed separately
- Sensitivity tables show hyperparameter impact (e.g., which LR values work best)
- Success rate monitoring ensures we catch systematic failures early

---

**Generated:** 2026-04-13  
**Python Analysis Script:** `scripts/analyze_grid_results.py`
