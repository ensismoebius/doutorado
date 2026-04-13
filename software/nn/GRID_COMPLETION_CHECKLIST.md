# Grid Search Re-Run: Completion Checklist

## ✅ Setup Complete

- [x] Previous batch killed and cleaned
- [x] All 402 result files cleared for fresh run
- [x] Fresh batch launched: `20260413_000614`
- [x] 10-minute timeout per profile (600s) — INCREASED from 180s
- [x] 12 workers running in parallel
- [x] Reduced log level (warn) for faster execution
- [x] Thread caps applied (OMP=1, BLAS=1, MKL=1)
- [x] Current status: 13 active processes, 9 results completed

## ✅ Analysis Infrastructure Ready

### Scripts Created
- [x] `scripts/analyze_grid_results.py` — Main analysis engine
- [x] `scripts/run_analysis.sh` — Interactive runner wrapper
- [x] Both use only Python stdlib (no dependencies needed)

### Documentation Created
- [x] `GRID_SEARCH_STATUS.md` — Detailed execution reference
- [x] `ANALYSIS_QUICKREF.md` — One-page quick start
- [x] `GRID_RERUN_SUMMARY.md` — Complete setup summary
- [x] This checklist

### Output Structure (Will Be Created)
- [ ] `analysis/master_comparison.csv` (all 402 ranked)
- [ ] `analysis/comparison_audio-window.csv` (288 profiles)
- [ ] `analysis/comparison_fused-window.csv` (96 profiles)
- [ ] `analysis/comparison_eeg-window.csv` (18 profiles)
- [ ] `analysis/sensitivity_*.csv` (6 files for hyperparameter analysis)
- [ ] `analysis/summary_statistics.csv` (key metrics)

## 📊 Batch Execution Monitoring

### Current Progress
- **Completed:** 9 / 402 profiles (2.2%)
- **Active Workers:** 12 jobs running
- **Estimated Time:** ~16-22 hours from start (2026-04-13 00:06)
- **Estimated Completion:** ~16:00-22:00 UTC same day

### Check Status Commands
```bash
# Quick count
ls src/experiments/03/results/*grid* 2>/dev/null | wc -l

# Live job log
tail -20 logs/grid_runs/20260413_000614_joblog.tsv

# Check if done
test -f logs/grid_runs/20260413_000614_DONE && echo "COMPLETE"
```

## 🎯 When Batch Completes

### Step 1: Verify Completion
```bash
# Confirm done marker exists
test -f logs/grid_runs/20260413_000614_DONE && echo "✓ Batch complete"

# Verify all 402 results
ls src/experiments/03/results/*grid* | wc -l  # Should show 402
```

### Step 2: Generate Analysis Tables
```bash
# Option A (recommended)
python3 scripts/analyze_grid_results.py

# Option B (interactive with status messages)
bash scripts/run_analysis.sh
```

### Step 3: Review Results
```bash
# View top 20 configurations
head -21 analysis/master_comparison.csv | column -t -s,

# Quick modality summaries
for f in analysis/comparison_*.csv; do
  echo "=== $(basename $f) ===" && head -6 $f | tail -5 && echo ""
done
```

### Step 4: Export & Archive
```bash
# Open in spreadsheet
libreoffice analysis/master_comparison.csv

# Save to git
git add analysis/ GRID_*.md ANALYSIS_*.md
git commit -m "Grid search 20260413: 402 profiles complete"

# Create backup
tar -czf snn_grid_20260413.tar.gz analysis/ logs/grid_runs/20260413_*
```

## 📈 Expected Output Format

### Master Comparison Table
```
Rank | Profile | Modality | Loss Type | LR | BS | HS | LS | 
Final Train Loss | Final Val Loss | Mean Val Loss | Best Val Loss | Epochs | Status
1    | audio-window_grid_001 | audio-window | mae | 0.0001 | 8 | 32 | 16 | 
0.7785 | 0.7790 | 0.7791 | 0.7779 | 8 | Success
```

### Sensitivity Analysis (Example)
```
Hyperparameter | Value | Avg Val Loss | Best Val Loss | Count
Learning Rate  | 0.0001 | 0.7801 | 0.7779 | 72
Learning Rate  | 0.0005 | 0.7825 | 0.7810 | 72
Learning Rate  | 0.001  | 0.7950 | 0.7890 | 72
Batch Size     | 4      | 0.7799 | 0.7778 | 43
Batch Size     | 8      | 0.7802 | 0.7779 | 43
...
```

## 🔍 Troubleshooting

### If no results generated after script runs:
1. Check: `ls src/experiments/03/results/` (should have files)
2. If empty, batch may still be running
3. Run: `tail -30 logs/grid_runs/20260413_000614_joblog.tsv | grep -i error`

### If script crashes:
1. Ensure Python 3.8+: `python3 --version`
2. Check JSON file validity: `head src/experiments/03/results/*grid_*.json`
3. Manually inspect one result: `cat src/experiments/03/results/audio-*.json | python3 -m json.tool`

### If batch appears hung:
1. Check job status: `ps aux | grep experiment03 | wc -l`
2. Check recent activity: `tail -5 logs/grid_runs/20260413_000614_joblog.tsv`
3. If stuck >600s on one job, check system load: `top -bn1 | head -5`

## 📝 Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `src/experiments/03/results/*grid*.json` | Raw result data | ✅ Generated |
| `logs/grid_runs/20260413_000614_joblog.tsv` | Job tracking | ✅ Live updates |
| `scripts/analyze_grid_results.py` | Analysis engine | ✅ Ready |
| `scripts/run_analysis.sh` | Interactive runner | ✅ Ready |
| `GRID_SEARCH_STATUS.md` | Reference docs | ✅ Created |
| `ANALYSIS_QUICKREF.md` | Quick start | ✅ Created |
| `GRID_RERUN_SUMMARY.md` | Setup summary | ✅ Created |
| `analysis/*.csv` | Output tables | ⏳ Pending |

## 🚀 One-Line Summary

**Full 402-profile SNN hyperparameter grid started at 00:06 UTC with 10-minute timeout per profile, expected to complete in 16-22 hours with comprehensive CSV analysis tables (master ranking, per-modality comparisons, hyperparameter sensitivity) ready via `python3 scripts/analyze_grid_results.py`**

---

**Status Updated:** 2026-04-13 00:15 UTC  
**Batch Status:** ✅ RUNNING HEALTHY
