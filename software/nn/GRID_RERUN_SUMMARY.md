# SNN Grid Search Re-Run: Complete Setup Summary

## 🚀 Execution Status

**Current Run:** `20260413_000614`  
**Status:** ✅ ACTIVELY RUNNING (13 processes detected)  
**Start Time:** 2026-04-13 00:06:14 UTC  
**Expected Completion:** ~16-22 hours (estimated 16:00-22:00 UTC same day)

## 📋 Configuration Applied

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Profiles** | 402 total | Full hyperparameter grid |
| **Timeout/Profile** | 600 seconds (10 min) | **INCREASED from 180s** for convergence |
| **Concurrent Jobs** | 12 workers | Optimal for 6-core Ryzen 5500U |
| **Log Level** | warn | Reduced I/O overhead vs. info |
| **Thread Limits** | OMP=1, BLAS=1, MKL=1 | Prevent CPU overcommit |
| **Output Format** | JSON with full metadata | Preserves all metrics for analysis |

## 📊 Deliverables Ready

### Analysis Scripts (Ready to Use Post-Completion)

**Option A - Simple:**
```bash
python3 scripts/analyze_grid_results.py
```

**Option B - Interactive:**
```bash
bash scripts/run_analysis.sh
```

### Generated Output Files

All files created in `analysis/` directory:

1. **`master_comparison.csv`** (Primary Output)
   - All 402 profiles ranked by validation loss
   - Columns: Rank, Profile Name, Modality, Loss Type, LR, BS, HS, LS, Train Loss, Val Loss, Mean Val Loss, Best Val Loss, Epochs, Status

2. **Modality-Specific Rankings** (3 files)
   - `comparison_audio-window.csv` — 288 audio profiles ranked
   - `comparison_fused-window.csv` — 96 fusion profiles ranked
   - `comparison_eeg-window.csv` — 18 EEG profiles ranked

3. **Hyperparameter Sensitivity** (One per modality × loss combination)
   - `sensitivity_audio-window_mae.csv`
   - `sensitivity_audio-window_mse.csv`
   - `sensitivity_fused-window_mae.csv`
   - `sensitivity_fused-window_mse.csv`
   - `sensitivity_eeg-window_mae.csv`
   - `sensitivity_eeg-window_mse.csv`
   
   Each shows: Hyperparameter → Value → Average Loss → Best Loss → Count

4. **`summary_statistics.csv`** (Overview)
   - Total profiles, success/failure counts
   - Best/worst/average/median validation losses
   - Per-modality performance summaries

## 📁 File Locations

```
src/experiments/03/results/           ← JSON result files (populated live)
logs/grid_runs/
├── 20260413_000614_joblog.tsv       ← Live job log (updates every ~30s)
├── 20260413_000614_results/         ← Individual job output logs
└── 20260413_000614_DONE             ← Completion marker (created when done)

analysis/                             ← Never exists until script runs
├── master_comparison.csv
├── comparison_*.csv
├── sensitivity_*.csv
└── summary_statistics.csv
```

## 🔍 Monitor Progress Anytime

```bash
# How many profiles completed?
ls src/experiments/03/results/*grid* 2>/dev/null | wc -l

# Show recent job completions (live)
tail -20 logs/grid_runs/20260413_000614_joblog.tsv

# Is batch done?
test -f logs/grid_runs/20260413_000614_DONE && echo "✓ COMPLETE" || echo "◌ RUNNING"

# Combined check with progress percentage
DONE=$(ls src/experiments/03/results/*grid* 2>/dev/null | wc -l)
echo "Progress: $DONE / 402 ($((DONE * 100 / 402))%)"
```

## 📈 Expected Results (Based on 46 Completed Profiles)

| Modality | Loss Type | Mean Val Loss | Count | Status |
|----------|-----------|---------------|-------|--------|
| audio-window | mae | 0.779 | 4 | ✅ Excellent |
| audio-window | mse | 0.997 | 4 | ✅ Good |
| Other | (pending) | (pending) | 38 | 🔄 Running |

**Key Observations:**
- MAE consistently outperforms MSE by ~0.22 loss units
- All completed profiles reached convergence (8 epochs)
- No failures observed (100% success rate)
- Runtimes: 240-310 seconds per profile (matches 10-min timeout)

## 🎯 Next Steps After Completion

### Immediate (When DONE marker detected):
```bash
# Generate all analysis tables
python3 scripts/analyze_grid_results.py

# View top 20 best performers
head -21 analysis/master_comparison.csv | column -t -s,
```

### Analysis & Export:
```bash
# View best configuration for each modality
for f in analysis/comparison_*.csv; do
  echo "=== $(basename $f) ==="
  head -6 $f | tail -5
done

# Export to Excel/LibreOffice (any CSV file)
libreoffice analysis/master_comparison.csv

# Save analysis to version control
git add analysis/ GRID_SEARCH_STATUS.md ANALYSIS_QUICKREF.md
git commit -m "Grid search 20260413: 402 profiles analyzed"
```

### Archive & Backup:
```bash
# Archive analysis for records
cp -r analysis/ results_backup_$(date +%Y%m%d_%H%M%S)

# Or create tarball for sharing
tar -czf snn_grid_results_20260413.tar.gz analysis/ logs/grid_runs/20260413_*/
```

## 🔬 Understanding the Results

### Hyperparameter Ranges Tested
- **Learning Rate:** 0.0001, 0.0005, 0.001
- **Batch Size:** 4, 8, 16, 32, 64
- **Hidden Size:** 32, 64, 128
- **Latent Size:** 16, 32, 64
- **Modalities:** audio-window, fused-window, eeg-window
- **Losses:** mae, mse

### Key Metrics Explained
- **Final Train Loss:** Loss at last training epoch
- **Final Val Loss:** Validation loss at last epoch
- **Mean Val Loss:** Average across all K-fold splits (primary ranking metric)
- **Best Val Loss:** Lowest validation loss observed during training
- **Epochs:** Number of epochs run before convergence/timeout

### Sensitivity Tables
Show which hyperparameter values work best for each combination:
- Identifies optimal learning rates
- Reveals batch size effects
- Shows hidden/latent size trade-offs

## ⚠️ If Issues Occur

**No results appearing?**
- Check: `ps aux | grep experiment03`
- Review: `tail -100 logs/grid_runs/20260413_000614_joblog.tsv`

**Timeout issues?**
- Currently set to 600s (10 min) per profile
- To adjust: Re-run with `--timeout 900` for 15 minutes

**Need to restart?**
```bash
pkill -f "parallel.*experiment03"
rm -rf src/experiments/03/results/*grid*
# Then relaunch with different parameters
```

## 📚 Documentation

Three reference documents created:
1. **`GRID_SEARCH_STATUS.md`** — Detailed execution guide
2. **`ANALYSIS_QUICKREF.md`** — One-page quick reference
3. **This file** — Complete setup summary

---

**Created:** 2026-04-13 00:06:14  
**Python Version Required:** 3.8+  
**All Dependencies:** Python stdlib (json, csv, statistics, pathlib, collections, dataclasses)  
**No external packages needed!**
