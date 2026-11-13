# autoencoderRunner Grid Runbook

How to sweep the [autoencoderRunner](../Experiments/AutoencoderRunner.md) SNN autoencoders over a
hyperparameter grid, and rank the results. This is the old "Experiment03" grid. It is not
part of the thesis or the paper (they have their own pipelines), and it is large, so read
§ Size before starting it.

## The one command

The wrapper does all three steps: it generates the profiles, runs them in parallel (GNU
`parallel`, one process per profile, with a joblog), and analyses the results. Every path is
required:

```bash
cd software/nn
cmake --build out/build/max-performance --target autoencoderRunner -j$(nproc)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NN_AUTOENCODERRUNNER_LOG_LEVEL=warn
bash src/experiments/autoencoderRunner/scripts/run_full_grid_and_analyze.sh \
  --project-root "$PWD" \
  --bin out/build/max-performance/src/experiments/autoencoderRunner/autoencoderRunner \
  --profiles-output-dir /scratch/grid/profiles \
  --results-dir "$PWD/results" \
  --analysis-output-dir /scratch/grid/analysis \
  --jobs 12 --timeout 600
```

Each profile reads `~/database.sqlite` (the Coretto base, see [Data Bases](./Data-Bases.md)).
The runner writes each summary (`<timestamp>_<profile>.json`) to `results/` under its working
directory, which the wrapper sets to `--project-root`. So `--results-dir` must be
`<project-root>/results`, and the wrapper refuses anything else. Summaries accumulate there,
and the analyzer reads every `*grid_*.json` at the top of that folder, earlier sweeps
included. Move those away before a new sweep. The thesis results sit in subfolders
(`results/thesis/`) and are not read. The joblog and the `…_DONE` marker go to
`logs/grid_runs/`.

## The steps, one by one

| Step | Command | Writes |
|---|---|---|
| 1. profiles | `python3 src/experiments/autoencoderRunner/scripts/create_test_profiles.py --output-dir <dir>` | one JSON per grid point (`<type>_grid_<n>_lr…_bs…_hs…_ls…_<loss>_mb…_ep…_lp….json`) |
| 2. runs | `parallel -j 12 --timeout 600 --joblog <log> "<bin> --profile {}" ::: <dir>/*.json` | `results/<timestamp>_<profile>.json`, under the working directory |
| 3. analysis | `python3 src/experiments/autoencoderRunner/scripts/analyze_grid_results.py --results-dir <results> --output-dir <analysis>` | `master_comparison.csv`, `comparison_<modality>.csv`, `sensitivity_<modality>_<loss>.csv`, `summary_statistics.csv` |

Both Python scripts take every directory explicitly and refuse a mistyped option. The analyzer
exits non-zero when the results directory is missing or holds no `*grid_*.json`, so an
empty analysis cannot pass for a finished one.

## Size

The grid is the full cross product of the axes in `param_grids` at the top of
`create_test_profiles.py` (learning rates, batch sizes, hidden and latent sizes, losses,
depths, …, per model type). As written it generates **97,280 profiles**: 30,720 audio-window,
30,720 EEG-window and 35,840 fused-window (counted on 2026-11-13). At a 600 s timeout and 12
parallel jobs that is weeks of machine time. Narrow `param_grids` before running, and count
with `ls <dir> | wc -l` after step 1.

## Troubleshooting

| Symptom | Check |
|---|---|
| the analyzer stops with "no *grid_*.json results" | the runs wrote somewhere else, or all failed: read the joblog's `Exitval` column |
| a run stalls | the `--timeout`, and the process count against the cores (`OMP_NUM_THREADS=1` per process) |
| a profile is refused | `python3 -m json.tool <profile>`; the runner refuses unknown keys |

## See Also

- [AutoencoderRunner](../Experiments/AutoencoderRunner.md)
