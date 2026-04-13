# Quick Reference: Grid Search Analysis & Results

## Current Status
- **Run ID:** 20260413_000614
- **Start Time:** 2026-04-13 00:06:14 
- **Status:** ✅ RUNNING (13 active processes)
- **Completion Marker:** Until `logs/grid_runs/20260413_000614_DONE` exists, batch is still running

## One-Command Results Analysis

**When batch finishes**, run this single command to generate all comparison tables:

```bash
python3 scripts/analyze_grid_results.py
```

This generates:
- `analysis/master_comparison.csv` — All 402 profiles ranked
- `analysis/comparison_audio-window.csv` — Top audio configurations
- `analysis/comparison_fused-window.csv` — Top fusion configurations  
- `analysis/comparison_eeg-window.csv` — Top EEG configurations
- `analysis/sensitivity_*.csv` — Hyperparameter impact tables
- `analysis/summary_statistics.csv` — Success rates & key metrics

## Check Progress (Anytime)

```bash
# How many done?
ls src/experiments/03/results/*grid* 2>/dev/null | wc -l

# How many total jobs?
grep -c "^[0-9]" logs/grid_runs/20260413_000614_joblog.tsv

# Still running?
test -f logs/grid_runs/20260413_000614_DONE && echo "DONE" || echo "RUNNING"

# Quick stats
echo "Completed: $(ls src/experiments/03/results/*grid* 2>/dev/null | wc -l)/402"
```

## Results CSV Columns

### Master Table (`master_comparison.csv`)
```
Rank | Profile | Modality | Loss Type | LR | BS | HS | LS | 
Final Train Loss | Final Val Loss | Mean Val Loss | Best Val Loss | Epochs | Status | Error
```

- **Mean Val Loss:** Primary ranking metric (lower = better)
- **Loss Type:** mae or mse
- **LR:** Learning rate (0.0001–0.001)
- **BS:** Batch size (4–64)
- **HS:** Hidden size (32–128)
- **LS:** Latent size (16–64)

### Sensitivity Tables (`sensitivity_*.csv`)
```
Hyperparameter | Value | Avg Val Loss | Best Val Loss | Count
```
Shows average performance for each setting. Use to identify optimal ranges.

## File Locations
- **Results:** `src/experiments/03/results/` (JSON format, one per profile)
- **Job Log:** `logs/grid_runs/20260413_000614_joblog.tsv` (live updates)
- **Analysis Output:** `analysis/` (appears after script runs)
- **Config Status:** `GRID_SEARCH_STATUS.md` (detailed execution log)

## Expected Outcomes

Based on 46 profiles from previous run:
- **Audio-Window MAE:** ~0.779 mean validation loss ✨
- **Audio-Window MSE:** ~0.997 mean validation loss 
- **Success Rate:** ~100% (no failures observed)
- **Avg Runtime:** 240-310 seconds per profile
- **Total Time:** ~16-22 hours for full batch

## After Results Arrive

1. **Generate tables:** `python3 scripts/analyze_grid_results.py`
2. **View best configs:** `head -20 analysis/master_comparison.csv`
3. **Export to Excel:** Copy any CSV to spreadsheet application
4. **Archive:** `cp -r analysis/ results_20260413_` for backup
5. **Commit:** `git add analysis/ CHANGELOG.md && git commit -m "Grid search 20260413: 402 profiles completed"`

## Hyperparameter Ranges in Grid

| Parameter | Values | Count |
|-----------|--------|-------|
| Learning Rate | 0.0001, 0.0005, 0.001 | 3 |
| Batch Size | 4, 8, 16, 32, 64 | 5 |
| Hidden Size | 32, 64, 128 | 3 |
| Latent Size | 16, 32, 64 | 3 |
| Modalities | audio-window, fused-window, eeg-window | 3 |
| Loss Types | mae, mse | 2 |

**Total:** 3 × 5 × 3 × 3 × 3 × 2 = 2,700 theoretical  
**Actually run:** 402 (filtered by experimental design)

## Top Performers (From 46 Completed)

| Rank | Profile | Loss Type | Mean Val Loss |
|------|---------|-----------|---------------|
| 1 | audio-window_grid_001 | mae | 0.779075 |
| 2 | audio-window_grid_003 | mae | 0.779044 |
| 3 | audio-window_grid_002 | mse | 0.996955 |
| 4 | audio-window_grid_000 | mse | 0.996657 |

*Note: Full ranking available once all 402 complete*

---

**Script Version:** 1.0  
**Python Version (Required):** 3.8+  
**Dependencies:** json, csv, statistics (all stdlib)
