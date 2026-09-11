#!/usr/bin/env bash
# 01_meeting01_run_loso.sh — nested leave-one-group-out run for the paper revision.
#
# Runs profiles/meeting01-loso.json once per (dataset, outer fold): --dataset {fsdd |
# audiomnist | mitbih} --cv-fold 0..N-1, each as its own process. Per run the binary:
# trains the LSTM-/GRU-/Transformer-AE baselines (fit on train, early-stop on val,
# evaluate once on val and once on the held-out test group); runs the SNN
# v_th x alpha x architecture sweep selecting on the inner validation group only, then
# retrains the winner on train + val (early-stopping on a recording-disjoint monitor
# carved from train) and evaluates it once on test.
#
# Outputs, per dataset d and fold f:
#   results/meeting01/meeting01_loso_<d>_fold<f>_comparative_metrics.csv   (split=val | test rows)
#   results/meeting01/meeting01_loso_<d>_fold<f>_per_window_errors.csv     (recording-level stats input)
#   results/meeting01/meeting01_loso_<d>_fold<f>_split_manifest.json       (leakage audit)
#   results/meeting01/meeting01_loso_<d>_fold<f>_<enc>_run<r>_model_selection_manifest.json
#
# COST: weeks over all datasets x folds even with the stratified per-fold window caps
#   (loso_max_{train,val,test}_windows in the profile: 1200 / 300 / 1500). Full pooled
#   FSDD is 27k train windows/fold — intractable across the 27-combo SNN grid x 3
#   encodings x 5 seeds x 18 (dataset,fold) processes. The caps keep every speaker and
#   recording represented (round-robin subsample); LOSO structure and the recording-level
#   statistical unit are unchanged. Per-epoch progress is logged as "[loso] ... epoch N/M"
#   lines (stderr; survives nohup, where the live bars collapse). Run once — by default.
#   This script REFUSES to start without EXPERIMENT_CONFIRMED=1, and clears
#   results/meeting01/checkpoints/ first (resumed rows are not regenerated into the
#   per-window CSV) UNLESS RESUME=1 (see below).
#
# Usage:
#   cd software/nn
#   EXPERIMENT_CONFIRMED=1 MEETING01_BUILD=max-performance \
#     ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
#
#   SKIP_BUILD=1   reuse the existing binary (only when you know it is current)
#   DATASETS      space-separated dataset list (default "fsdd audiomnist mitbih")
#   CV_NUM_FOLDS   number of outer folds (default 6)
#   KEEP_CHECKPOINTS=1  do not clear checkpoints (per-window CSV will then be incomplete
#                       for any fold re-run this way — RESUME=1 is almost always what
#                       you actually want instead, see below)
#
#   RESUME=1   the simple way to continue an interrupted run: re-invoke the exact same
#              command line with RESUME=1 added. It implies KEEP_CHECKPOINTS=1 (nothing
#              is wiped), and before each (dataset, fold) skips straight past any pair
#              that already has a complete `*_comparative_metrics.csv` — that file is
#              written exactly once, at the very end of a fold's run, so its presence
#              means the fold genuinely finished, not that it merely started. Only the
#              fold that was actually running when the script died (no CSV yet) and
#              everything after it get (re)run.
#                cd software/nn
#                EXPERIMENT_CONFIRMED=1 RESUME=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

REFERENCE_BUILD="max-performance"
MEETING01_BUILD="${MEETING01_BUILD:-$REFERENCE_BUILD}"
CV_NUM_FOLDS="${CV_NUM_FOLDS:-6}"
PROFILE="src/experiments/meeting01/profiles/meeting01-loso.json"
BIN="out/build/${MEETING01_BUILD}/src/experiments/meeting01/meeting01"

if [[ "${EXPERIMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "REFUSED: nested-LOSO run is days-scale. Re-invoke with EXPERIMENT_CONFIRMED=1." >&2
  exit 1
fi

if [[ "$MEETING01_BUILD" != "$REFERENCE_BUILD" ]]; then
  echo "[loso-run] ⚠  '${MEETING01_BUILD}' is NOT the reference backend (${REFERENCE_BUILD})."
  echo "[loso-run] ⚠  train_ms / infer_ms feed the paper directly — report the backend used."
fi

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "[loso-run] SKIP_BUILD=1 but no binary at $BIN" >&2; exit 1; }
  echo "[loso-run] SKIP_BUILD=1 — reusing $BIN"
else
  echo "[loso-run] configuring/building ${MEETING01_BUILD}"
  cmake --preset="${MEETING01_BUILD}"
  cmake --build --preset="${MEETING01_BUILD}" -j"$(nproc)" --target meeting01
fi
[[ -x "$BIN" ]] || { echo "[loso-run] no binary at $BIN after build" >&2; exit 1; }

RESUME="${RESUME:-0}"
if [[ "$RESUME" == "1" ]]; then
  KEEP_CHECKPOINTS=1
  echo "[loso-run] RESUME=1 — keeping checkpoints/models/events, skipping already-finished (dataset, fold) pairs"
fi

if [[ "${KEEP_CHECKPOINTS:-0}" != "1" ]]; then
  echo "[loso-run] clearing results/meeting01/{checkpoints,models}/ and stale *_events.jsonl"
  rm -rf results/meeting01/checkpoints/ results/meeting01/models/
  rm -f results/meeting01/*_events.jsonl
fi

# Recorded in every session_begin event so the live monitor can show provenance.
export MEETING01_GIT_COMMIT="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "[loso-run] live dashboard (separate terminal, from software/nn):"
echo "[loso-run]   ${ROOT_DIR}/.venv/bin/python3 scripts/pipeline/meeting01/monitor.py --run-tag meeting01_loso"
echo "[loso-run]   (add --plain when piped / not a terminal; --rank N for one config's detail)"

DATASETS="${DATASETS:-fsdd audiomnist mitbih}"
_start=$(date +%s)
_nds=$(wc -w <<< "$DATASETS")
_di=0
for ds in $DATASETS; do
  _di=$((_di + 1))
  for (( f = 0; f < CV_NUM_FOLDS; f++ )); do
    _f_start=$(date +%s)
    _fold_csv="results/meeting01/meeting01_loso_${ds}_fold${f}_comparative_metrics.csv"
    if [[ "$RESUME" == "1" && -s "$_fold_csv" ]]; then
      echo "[loso-run] === ${ds} fold ${f}/${CV_NUM_FOLDS} === already complete ($_fold_csv exists) — RESUME=1 skip"
      continue
    fi
    export MEETING01_OVERALL="$(printf 'LOSO  %s (%d/%d)  fold %d/%d  elapsed %s' \
      "$ds" "$_di" "$_nds" "$((f + 1))" "$CV_NUM_FOLDS" "$(( $(date +%s) - _start ))s")"
    echo "[loso-run] === ${ds} fold ${f}/${CV_NUM_FOLDS} ==="
    "$BIN" --comparative-config "$PROFILE" --dataset "$ds" --cv-fold "$f"
    echo "[loso-run] ${ds} fold ${f} done in $(( $(date +%s) - _f_start ))s"
  done
done
unset MEETING01_OVERALL
printf '[loso-run] all %d datasets x %d folds done in %ss\n' \
  "$_nds" "$CV_NUM_FOLDS" "$(( $(date +%s) - _start ))"

PY="python3"
[[ -x "$ROOT_DIR/.venv/bin/python3" ]] && PY="$ROOT_DIR/.venv/bin/python3"
PAPER_DATA="/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/meeting01/data"

# Order matters: 03_ appends the pca/mean rows that 02_ and 04_ then read.
echo "[loso-run] fitting PCA / mean-frame reference baselines (per fold, train-only)"
"$PY" scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py \
  --results-dir results/meeting01 --run-tag meeting01_loso --latent 32

echo "[loso-run] aggregating per-fold test rows into paper tables (mean +/- std over seeds)"
"$PY" scripts/pipeline/meeting01/02_meeting01_build_loso_paper_data.py \
  --results-dir results/meeting01 --run-tag meeting01_loso --data-dir "$PAPER_DATA"

echo "[loso-run] hierarchical significance analysis (recording-level primary)"
"$PY" scripts/pipeline/meeting01/04_meeting01_significance_tests.py \
  --results-dir results/meeting01 --run-tag meeting01_loso --out-dir "$PAPER_DATA"

echo "[loso-run] done — per-fold CSVs, per_window_errors.csv, paper_loso_*.csv, *_significance.json ready"
