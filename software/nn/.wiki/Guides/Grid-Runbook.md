# SNN Grid Tests Comprehensive Runbook

This document is the canonical guide for running, monitoring, analyzing, and archiving SNN grid tests.

## Quickstart (5 Commands)

```bash
cd <path_to_nn_project_root>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NN_EXPERIMENT03_LOG_LEVEL=warn
run_id="$(date +%Y%m%d_%H%M%S)" && mkdir -p logs/grid_runs
parallel -j 12 --timeout 600 --joblog "logs/grid_runs/${run_id}_joblog.tsv" "./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03 --profile {}" ::: src/experiments/03/profiles/snnAutoEncodersProfiles/*.json && touch "logs/grid_runs/${run_id}_DONE"
python3 src/experiments/03/scripts/analyze_grid_results.py
```

## 1. Scope and Goal

- Re-run profiles with more time per profile (600s timeout)
- Generate comprehensive comparison tables for all tested configurations

## 2. Grid Definition

Total profiles: 402
- audio-window: 288
- fused-window: 96
- eeg-window: 18

### Hyperparameters Tested
- Learning rate: 0.0001, 0.0005, 0.001
- Batch size: 4, 8, 16, 32, 64
- Hidden size: 32, 64, 128
- Latent size: 16, 32, 64

## 3. Environment

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NN_EXPERIMENT03_LOG_LEVEL=warn
```

## 4. Run Commands

### Parallel (Recommended)
```bash
cd <path_to_nn_project_root>
bin="./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03"
mkdir -p logs/grid_runs
run_id="$(date +%Y%m%d_%H%M%S)"

parallel -j 12 \
  --timeout 600 \
  --joblog "logs/grid_runs/${run_id}_joblog.tsv" \
  "$bin --profile {}" \
  ::: src/experiments/03/profiles/snnAutoEncodersProfiles/*.json
```

## 5. Analysis

```bash
python3 src/experiments/03/scripts/analyze_grid_results.py
```

Output:
- `analysis/master_comparison.csv` - Full ranked results
- `analysis/comparison_<modality>.csv` - Per-modality breakdowns
- `analysis/sensitivity_<modality>_<loss>.csv` - Hyperparameter sensitivity
- `analysis/summary_statistics.csv` - Aggregate statistics

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| No results | Check JSON outputs in `src/experiments/03/results/` |
| Stalled batch | Check process count and joblog |
| Script failure | Verify Python 3.8+, validate JSON with `python3 -m json.tool` |

## See Also

- [Experiment03](../Experiments/Experiment03.md)
- [Architecture](../Architecture.md)