# SNN Grid Tests Comprehensive Runbook

Generated: 2026-04-13

This document is the single canonical guide for running, monitoring, analyzing, and archiving the SNN grid tests. It merges and supersedes the previous operational docs.

## Quickstart (5 Commands)

Run these commands in order for fastest end-to-end execution:

```bash
cd <path_to_nn_project_root>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NN_EXPERIMENT03_LOG_LEVEL=warn
run_id="$(date +%Y%m%d_%H%M%S)" && mkdir -p logs/grid_runs
parallel -j 12 --timeout 600 --joblog "logs/grid_runs/${run_id}_joblog.tsv" "./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03 --profile {}" ::: src/experiments/03/profiles/snnAutoEncodersProfiles/*.json && touch "logs/grid_runs/${run_id}_DONE"
python3 src/experiments/03/scripts/analyze_grid_results.py
```

Immediate validation:

```bash
wc -l analysis/master_comparison.csv
head -21 analysis/master_comparison.csv | column -t -s,
```

## 1. Scope and Goal

Requested objective:
- Re-run profiles with more time per profile.
- Generate comprehensive comparison tables containing all tested data and results for future study.

Delivered approach:
- Per-profile timeout increased from 180s to 600s.
- Full grid execution pipeline defined for 402 profiles.
- Analysis tooling produces complete ranking, per-modality breakdowns, sensitivity tables, and summary statistics.

## 2. Grid Definition

Total profiles: 402
- audio-window: 288
- fused-window: 96
- eeg-window: 18

Modalities:
- audio-window
- fused-window
- eeg-window

Losses:
- mae
- mse

Primary hyperparameters:
- Learning rate (LR): 0.0001, 0.0005, 0.001
- Batch size (BS): 4, 8, 16, 32, 64
- Hidden size (HS): 32, 64, 128
- Latent size (LS): 16, 32, 64

Common training settings:
- training_epochs: 8
- training_max_batches_per_epoch: 30
- training_normalize_inputs: true
- training_lr_plateau_enabled: true
- training_lr_plateau_factor: 0.5
- training_lr_plateau_patience: 2
- training_lr_plateau_min_delta: 0.000001
- kfold_enabled: false
- sampler_shuffle_seed: 42

## 3. Environment and Runtime Configuration

Project root:
- /home/ensismoebius/Repos/doutorado/software/nn

Binary used in examples:
- ./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03

Recommended environment:
- OMP_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1
- MKL_NUM_THREADS=1
- NN_EXPERIMENT03_LOG_LEVEL=warn

Execution profile:
- 12 concurrent workers
- 600s timeout per profile

## 4. Run Commands

### 4.1 Parallel run (recommended)

```bash
cd <path_to_nn_project_root>

bin="./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NN_EXPERIMENT03_LOG_LEVEL=warn

mkdir -p logs/grid_runs
run_id="$(date +%Y%m%d_%H%M%S)"

parallel -j 12 \
  --timeout 600 \
  --joblog "logs/grid_runs/${run_id}_joblog.tsv" \
  "$bin --profile {}" \
  ::: src/experiments/03/profiles/snnAutoEncodersProfiles/*.json

echo "Run complete: ${run_id}"
touch "logs/grid_runs/${run_id}_DONE"
```

### 4.2 Sequential run

```bash
cd <path_to_nn_project_root>

bin="./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NN_EXPERIMENT03_LOG_LEVEL=warn

for profile in src/experiments/03/profiles/snnAutoEncodersProfiles/*.json; do
  timeout 600 "$bin" --profile "$profile"
done
```

## 5. Live Monitoring

```bash
# Completed results count
ls src/experiments/03/results/*grid*.json 2>/dev/null | wc -l

# Recent jobs
tail -20 logs/grid_runs/<RUN_ID>_joblog.tsv

# Process count (rough)
ps aux | grep experiment03 | wc -l

# Done marker check
test -f logs/grid_runs/<RUN_ID>_DONE && echo "COMPLETE" || echo "RUNNING"
```

Notes:
- Typical profile runtime observed in prior runs: about 177-310 seconds.
- Full 402-profile completion estimate: about 16-24 hours depending on system load.

## 6. Analysis Generation

Run analysis at any time (partial) or after completion (full):

```bash
python3 src/experiments/03/scripts/analyze_grid_results.py
```

Optional wrapper:

```bash
bash src/experiments/03/scripts/run_analysis.sh
```

No external Python dependencies required (stdlib only).

## 7. Analysis Outputs

Output directory:
- analysis/

Primary files:
- master_comparison.csv
- comparison_audio-window.csv
- comparison_fused-window.csv
- comparison_eeg-window.csv
- sensitivity_<modality>_<loss>.csv
- summary_statistics.csv

Expected primary deliverable size:
- master_comparison.csv around 30-40 KB for 402 rows plus header

## 8. Data Schema

### 8.1 master_comparison.csv columns

- Rank
- Profile
- Modality
- Loss Type
- LR
- BS
- HS
- LS
- Final Train Loss
- Final Val Loss
- Mean Val Loss
- Best Val Loss
- Epochs
- Status
- Error

Ranking criterion:
- Lower Mean Val Loss is better.

### 8.2 sensitivity_<modality>_<loss>.csv columns

- Hyperparameter
- Value
- Avg Val Loss
- Best Val Loss
- Count

### 8.3 summary_statistics.csv typical fields

- Total Profiles
- Successful
- Failed
- Success Rate %
- Best Mean Val Loss
- Worst Mean Val Loss
- Avg Mean Val Loss
- Median Mean Val Loss
- StdDev Mean Val Loss
- Per-modality best and count entries

## 9. Immediate Usage Commands

```bash
# Human-readable preview
head -21 analysis/master_comparison.csv | column -t -s,

# Open in spreadsheet
libreoffice analysis/master_comparison.csv

# Verify full expected row count after complete run (403 = header + 402 profiles)
wc -l analysis/master_comparison.csv
```

Quick Python exploration:

```bash
python3 << 'PYEOF'
import pandas as pd

df = pd.read_csv('analysis/master_comparison.csv')
print('Best configuration:')
print(df.iloc[0])

print('\nTop modality entries:')
for mod in df['Modality'].dropna().unique():
    top = df[df['Modality'] == mod].iloc[0]
    print(f"{mod}: {top['Mean Val Loss']:.6f}")
PYEOF
```

## 10. Interpretation Guide

Use the ranking and sensitivity outputs as follows:
- Rank 1 is best overall by Mean Val Loss.
- Differences smaller than about 0.001 are often practically tied.
- Compare MAE vs MSE within same modality and similar architecture.
- Use sensitivity tables to detect stable ranges (not only single best points).

Typical observed trend in prior partial runs:
- MAE often outperformed MSE on validation loss.

## 11. Completion Workflow

Recommended finalization steps:

```bash
# 1) Confirm run completion
test -f logs/grid_runs/<RUN_ID>_DONE && echo "Run done"

# 2) Regenerate all tables
python3 src/experiments/03/scripts/analyze_grid_results.py

# 3) Inspect key output
head -21 analysis/master_comparison.csv | column -t -s,

# 4) Archive analysis snapshot
tar -czf snn_grid_results_$(date +%Y%m%d_%H%M%S).tar.gz analysis/
```

## 12. Troubleshooting

No results generated:
- Confirm JSON outputs exist in src/experiments/03/results/.
- Inspect job log tail for failures.

Batch appears stalled:
- Check experiment03 process count.
- Check latest entries in job log.
- Confirm machine has not rebooted and run still active.

Script failure:
- Verify Python version is 3.8+.
- Validate a result JSON with python3 -m json.tool.

Need restart:

```bash
pkill -f "parallel.*experiment03"
rm -f src/experiments/03/results/*grid*.json
```

Then relaunch with Section 4 commands.

## 13. Optional Canonicalization for Multi-Run Histories

If multiple reruns produced duplicate profile signatures, keep a canonical deduplicated table for study:
- Raw aggregate: analysis/master_comparison.csv
- Canonical dedup aggregate: analysis/master_comparison_dedup_by_profile.csv

If dedup step is part of your workflow, use the dedup table as the reference dataset for downstream reporting.

## 14. Reproducibility and Recordkeeping

Best practice checklist:
- Keep run_id in logs/grid_runs.
- Keep joblog.tsv and DONE marker.
- Keep raw JSON result files for provenance.
- Regenerate analysis tables from raw JSON when needed.
- Archive analysis outputs with timestamped tarball.

## 15. Single-Page Checklist

- Configure environment thread caps and warn logging.
- Run profiles with 600s timeout.
- Monitor progress via results count and joblog.
- Wait for DONE marker.
- Run python3 src/experiments/03/scripts/analyze_grid_results.py.
- Inspect master_comparison.csv.
- Archive analysis outputs.

This runbook is now the single source of truth for executing and analyzing these grid tests.