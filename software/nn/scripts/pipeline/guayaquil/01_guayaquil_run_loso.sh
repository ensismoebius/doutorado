#!/usr/bin/env bash
# 01_guayaquil_run_loso.sh — nested leave-one-speaker-out run for the paper revision.
#
# Runs profiles/article-loso.json once per outer fold (--cv-fold 0..N-1), each as its
# own process. Per fold the binary: trains the LSTM-/GRU-/Transformer-AE baselines
# (fit on train, early-stop on val, evaluate once on val and once on the held-out test
# speaker); runs the SNN v_th x alpha x architecture sweep selecting on the inner
# validation speaker only, then retrains the winner on train + val (early-stopping on a
# recording-disjoint monitor carved from train) and evaluates it once on test.
#
# Outputs, per fold f:
#   results/guayaquil/article_loso_fold<f>_comparative_metrics.csv     (split=val | test rows)
#   results/guayaquil/article_loso_fold<f>_per_window_errors.csv       (recording-level stats input)
#   results/guayaquil/article_loso_fold<f>_split_manifest.json         (leakage audit)
#   results/guayaquil/article_loso_fold<f>_<enc>_run<r>_model_selection_manifest.json
#
# COST: days-scale over all folds. Run once. This script REFUSES to start without
#   EXPERIMENT_CONFIRMED=1, and clears results/guayaquil/checkpoints/ first (resumed
#   rows are not regenerated into the per-window CSV — see GuayaquilExperiment.cpp).
#
# Usage:
#   cd software/nn
#   EXPERIMENT_CONFIRMED=1 GUAYAQUIL_BUILD=max-performance \
#     ./scripts/pipeline/guayaquil/01_guayaquil_run_loso.sh
#
#   SKIP_BUILD=1   reuse the existing binary (only when you know it is current)
#   CV_NUM_FOLDS   number of outer folds (default 6, the FSDD speaker count)
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

_start=$(date +%s)
for (( f = 0; f < CV_NUM_FOLDS; f++ )); do
  _f_start=$(date +%s)
  export GUAYAQUIL_OVERALL="$(printf 'LOSO  fold %d/%d  elapsed %s' \
    "$((f + 1))" "$CV_NUM_FOLDS" "$(( $(date +%s) - _start ))s")"
  echo "[loso-run] === fold ${f}/${CV_NUM_FOLDS} ==="
  "$BIN" --comparative-config "$PROFILE" --cv-fold "$f"
  echo "[loso-run] fold ${f} done in $(( $(date +%s) - _f_start ))s"
done
unset GUAYAQUIL_OVERALL
printf '[loso-run] all %d folds done in %ss\n' "$CV_NUM_FOLDS" "$(( $(date +%s) - _start ))"

PY="python3"
[[ -x "$ROOT_DIR/.venv/bin/python3" ]] && PY="$ROOT_DIR/.venv/bin/python3"
PAPER_DATA="/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data"

echo "[loso-run] fitting PCA / mean-frame reference baselines (per fold, train-only)"
"$PY" scripts/pipeline/guayaquil/03_guayaquil_pca_mean_baselines.py \
  --results-dir results/guayaquil --run-tag article_loso --latent 32

echo "[loso-run] hierarchical significance analysis (recording-level primary)"
"$PY" scripts/pipeline/guayaquil/04_guayaquil_significance_tests.py \
  --results-dir results/guayaquil --run-tag article_loso --out-dir "$PAPER_DATA"

echo "[loso-run] done — raw per-fold CSVs + per_window_errors.csv + *_significance.json ready"
