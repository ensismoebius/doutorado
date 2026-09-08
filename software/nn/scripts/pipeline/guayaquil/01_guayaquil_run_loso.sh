#!/usr/bin/env bash
# 01_guayaquil_run_loso.sh — nested leave-one-group-out run for the paper revision.
#
# Runs profiles/article-loso.json once per (dataset, outer fold): --dataset {fsdd |
# audiomnist | mitbih} --cv-fold 0..N-1, each as its own process. Per run the binary:
# trains the LSTM-/GRU-/Transformer-AE baselines (fit on train, early-stop on val,
# evaluate once on val and once on the held-out test group); runs the SNN
# v_th x alpha x architecture sweep selecting on the inner validation group only, then
# retrains the winner on train + val (early-stopping on a recording-disjoint monitor
# carved from train) and evaluates it once on test.
#
# Outputs, per dataset d and fold f:
#   results/guayaquil/article_loso_<d>_fold<f>_comparative_metrics.csv   (split=val | test rows)
#   results/guayaquil/article_loso_<d>_fold<f>_per_window_errors.csv     (recording-level stats input)
#   results/guayaquil/article_loso_<d>_fold<f>_split_manifest.json       (leakage audit)
#   results/guayaquil/article_loso_<d>_fold<f>_<enc>_run<r>_model_selection_manifest.json
#
# COST: days-to-weeks over all datasets x folds. Run once. This script REFUSES to start
#   without EXPERIMENT_CONFIRMED=1, and clears results/guayaquil/checkpoints/ first
#   (resumed rows are not regenerated into the per-window CSV — see GuayaquilExperiment.cpp).
#
# Usage:
#   cd software/nn
#   EXPERIMENT_CONFIRMED=1 GUAYAQUIL_BUILD=max-performance \
#     ./scripts/pipeline/guayaquil/01_guayaquil_run_loso.sh
#
#   SKIP_BUILD=1   reuse the existing binary (only when you know it is current)
#   DATASETS      space-separated dataset list (default "fsdd audiomnist mitbih")
#   CV_NUM_FOLDS   number of outer folds (default 6)
#   KEEP_CHECKPOINTS=1  do not clear checkpoints (resume a partial run; per-window CSV
#                       will then be incomplete)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

REFERENCE_BUILD="max-performance"
GUAYAQUIL_BUILD="${GUAYAQUIL_BUILD:-$REFERENCE_BUILD}"
CV_NUM_FOLDS="${CV_NUM_FOLDS:-6}"
PROFILE="src/experiments/guayaquil/profiles/article-loso.json"
BIN="out/build/${GUAYAQUIL_BUILD}/src/experiments/guayaquil/guayaquil"

if [[ "${EXPERIMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "REFUSED: nested-LOSO run is days-scale. Re-invoke with EXPERIMENT_CONFIRMED=1." >&2
  exit 1
fi

if [[ "$GUAYAQUIL_BUILD" != "$REFERENCE_BUILD" ]]; then
  echo "[loso-run] ⚠  '${GUAYAQUIL_BUILD}' is NOT the reference backend (${REFERENCE_BUILD})."
  echo "[loso-run] ⚠  train_ms / infer_ms feed the paper directly — report the backend used."
fi

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "[loso-run] SKIP_BUILD=1 but no binary at $BIN" >&2; exit 1; }
  echo "[loso-run] SKIP_BUILD=1 — reusing $BIN"
else
  echo "[loso-run] configuring/building ${GUAYAQUIL_BUILD}"
  cmake --preset="${GUAYAQUIL_BUILD}"
  cmake --build --preset="${GUAYAQUIL_BUILD}" -j"$(nproc)" --target guayaquil
fi
[[ -x "$BIN" ]] || { echo "[loso-run] no binary at $BIN after build" >&2; exit 1; }

if [[ "${KEEP_CHECKPOINTS:-0}" != "1" ]]; then
  echo "[loso-run] clearing results/guayaquil/checkpoints/"
  rm -rf results/guayaquil/checkpoints/
fi

DATASETS="${DATASETS:-fsdd audiomnist mitbih}"
_start=$(date +%s)
_nds=$(wc -w <<< "$DATASETS")
_di=0
for ds in $DATASETS; do
  _di=$((_di + 1))
  for (( f = 0; f < CV_NUM_FOLDS; f++ )); do
    _f_start=$(date +%s)
    export GUAYAQUIL_OVERALL="$(printf 'LOSO  %s (%d/%d)  fold %d/%d  elapsed %s' \
      "$ds" "$_di" "$_nds" "$((f + 1))" "$CV_NUM_FOLDS" "$(( $(date +%s) - _start ))s")"
    echo "[loso-run] === ${ds} fold ${f}/${CV_NUM_FOLDS} ==="
    "$BIN" --comparative-config "$PROFILE" --dataset "$ds" --cv-fold "$f"
    echo "[loso-run] ${ds} fold ${f} done in $(( $(date +%s) - _f_start ))s"
  done
done
unset GUAYAQUIL_OVERALL
printf '[loso-run] all %d datasets x %d folds done in %ss\n' \
  "$_nds" "$CV_NUM_FOLDS" "$(( $(date +%s) - _start ))"

PY="python3"
[[ -x "$ROOT_DIR/.venv/bin/python3" ]] && PY="$ROOT_DIR/.venv/bin/python3"
PAPER_DATA="/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data"

# Order matters: 03_ appends the pca/mean rows that 02_ and 04_ then read.
echo "[loso-run] fitting PCA / mean-frame reference baselines (per fold, train-only)"
"$PY" scripts/pipeline/guayaquil/03_guayaquil_pca_mean_baselines.py \
  --results-dir results/guayaquil --run-tag article_loso --latent 32

echo "[loso-run] aggregating per-fold test rows into paper tables (mean +/- std over seeds)"
"$PY" scripts/pipeline/guayaquil/02_guayaquil_build_loso_paper_data.py \
  --results-dir results/guayaquil --run-tag article_loso --data-dir "$PAPER_DATA"

echo "[loso-run] hierarchical significance analysis (recording-level primary)"
"$PY" scripts/pipeline/guayaquil/04_guayaquil_significance_tests.py \
  --results-dir results/guayaquil --run-tag article_loso --out-dir "$PAPER_DATA"

echo "[loso-run] done — per-fold CSVs, per_window_errors.csv, paper_loso_*.csv, *_significance.json ready"
