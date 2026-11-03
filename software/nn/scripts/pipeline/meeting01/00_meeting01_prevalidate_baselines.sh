#!/usr/bin/env bash
# 00_meeting01_prevalidate_baselines.sh -- the evidence for removing the LSTM-AE, GRU-AE and
# Transformer-AE from the meeting01 grid (author's decision 2026-11-02).
#
# What it runs. profiles/meeting01-prevalidation.json: the grid's profile with every family
# back in and ONE reduced NSGA-II budget for all of them, on the development folds only
# (fsdd fold 0, eegmmidb fold 0 -- DEVELOPMENT_FOLDS, already excluded from every headline
# number). One process per (family, dataset):
#
#   family process     runs                         results in
#   snn-ae             SNN-AE search + ann-ae twin  results/meeting01_prevalidation/snn-ae/
#   lstm-ae            LSTM-AE search               results/meeting01_prevalidation/lstm-ae/
#   gru-ae             GRU-AE search                results/meeting01_prevalidation/gru-ae/
#   transformer-ae     Transformer-AE search        results/meeting01_prevalidation/transformer-ae/
#
# each with one compute thread (the program gets slower with more, see the GridUnesp wiki's
# "Threads"), JOBS of them at a time. Every process's wall time and exit code go to
# <family>/<dataset>_fold<f>_process.json, written when it ends. Then 03_ fits PCA / the mean
# frame on the snn-ae folds, and 00_meeting01_prevalidation_report.py writes
#   results/meeting01_prevalidation/prevalidation_summary.csv
#   <paper>/data/paper_prevalidation.tex, paper_prevalidation_config.tex
#
# COST: hours to days -- the sequence families dominate it (that is what it measures). It
#   REFUSES to start without EXPERIMENT_CONFIRMED=1 (and the expensive-experiment-guard hook
#   blocks it in Claude Code without the same marker).
#
# Usage (from software/nn):
#   EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/00_meeting01_prevalidate_baselines.sh
#   JOBS=N                 processes at a time (default 4; 8 runs every task at once)
#   RESUME=1               keep what finished: a task whose _process.json says exit 0 is
#                          skipped; any other starts OVER, its files moved to
#                          <family>/interrupted_<time>_<ds>_fold<f>/ (not continued from its GA
#                          checkpoint: its fold time must be one interval, see run loop below)
#   SKIP_BUILD=1           reuse the existing binary (it must be built from these sources)
#   SKIP_POSTPROCESS=1     stop after the eight tasks (GridUnesp: 03_ and the report run where
#                          the paper is, after the results are copied home)
#   PREVALIDATION_PROFILE  another profile (a smoke test); PREVALIDATION_DIR  another output dir
#   PAPER_DATA             where the paper's two files go (default: the paper's data/)
set -euo pipefail
export LC_ALL=C # date's "%s.%N" and awk below must agree on the decimal point

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

REFERENCE_BUILD="max-performance"
MEETING01_BUILD="${MEETING01_BUILD:-$REFERENCE_BUILD}"
PROFILE="${PREVALIDATION_PROFILE:-src/experiments/meeting01/profiles/meeting01-prevalidation.json}"
OUT_ROOT="${PREVALIDATION_DIR:-results/meeting01_prevalidation}"
PAPER_DATA="${PAPER_DATA:-/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/meeting01/data}"
JOBS="${JOBS:-4}"
RESUME="${RESUME:-0}"
BIN="out/build/${MEETING01_BUILD}/src/experiments/meeting01/meeting01"
SCRIPTS="scripts/pipeline/meeting01"

if [[ "${EXPERIMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "REFUSED: the pre-validation is hours-to-days scale. Re-invoke with EXPERIMENT_CONFIRMED=1." >&2
  exit 1
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "REFUSED: JOBS must be a positive integer (got '$JOBS')" >&2; exit 1; }
[[ -f "$PROFILE" ]] || { echo "REFUSED: no profile at $PROFILE" >&2; exit 1; }
if [[ "$RESUME" != "1" && -d "$OUT_ROOT" && -n "$(ls -A "$OUT_ROOT")" ]]; then
  echo "REFUSED: $OUT_ROOT is not empty. Move it aside for a fresh run, or RESUME=1 to continue it." >&2
  exit 1
fi

PY="python3"
[[ -x "$ROOT_DIR/.venv/bin/python3" ]] && PY="$ROOT_DIR/.venv/bin/python3"
# shellcheck source=./_provenance.sh
source "$ROOT_DIR/$SCRIPTS/_provenance.sh"
provenance_resolve "$ROOT_DIR" "$PY"

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "[prevalidation] SKIP_BUILD=1 but no binary at $BIN" >&2; exit 1; }
  provenance_check_binary_is_current "$ROOT_DIR" "$MEETING01_BUILD"
else
  echo "[prevalidation] configuring/building ${MEETING01_BUILD}"
  cmake --preset="${MEETING01_BUILD}"
  cmake --build --preset="${MEETING01_BUILD}" -j"$(nproc)" --target meeting01
  provenance_mark_built "$ROOT_DIR" "$MEETING01_BUILD"
fi
[[ -x "$BIN" ]] || { echo "[prevalidation] no binary at $BIN after build" >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "$OUT_ROOT"
echo "$(date '+%F %T') ${MEETING01_GIT_COMMIT} RESUME=${RESUME} JOBS=${JOBS} threads=${OMP_NUM_THREADS}/${OPENBLAS_NUM_THREADS} profile=${PROFILE}" \
  >> "$OUT_ROOT/source_revisions.log"
echo "[prevalidation] source revision ${MEETING01_GIT_COMMIT}; threads OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}; ${JOBS} at a time"

mapfile -t TASKS < <("$PY" "$SCRIPTS/00_meeting01_prevalidation_report.py" --plan \
  --profile "$PROFILE" --out-root "$OUT_ROOT")
((${#TASKS[@]} > 0)) || { echo "[prevalidation] the plan has no task" >&2; exit 1; }

# One task: the binary on one (family, dataset, fold), then its record, written through a
# rename so a record that exists is a whole one.
run_task() {
  local fam="$1" ds="$2" fold="$3" prof="$4"
  local log="$OUT_ROOT/$fam/${ds}_fold${fold}.log"
  local record="$OUT_ROOT/$fam/${ds}_fold${fold}_process.json"
  # date, not $EPOCHREALTIME: that is bash 5, and GridUnesp's AlmaLinux 8 has bash 4.4, where
  # under `set -u` it failed every task before the binary started (smoke job 5314472).
  local t0 rc=0
  t0="$(date +%s.%N)"
  "$BIN" --comparative-config "$prof" --dataset "$ds" --cv-fold "$fold" --no-tui >"$log" 2>&1 || rc=$?
  local wall
  wall="$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.3f", b - a }')"
  printf '{"family": "%s", "dataset": "%s", "fold": %d, "exit": %d, "wall_s": %s, "log": "%s", "source_revision": "%s", "omp_threads": "%s", "openblas_threads": "%s"}\n' \
    "$fam" "$ds" "$fold" "$rc" "$wall" "$log" "$MEETING01_GIT_COMMIT" "$OMP_NUM_THREADS" \
    "$OPENBLAS_NUM_THREADS" > "$record.tmp"
  mv "$record.tmp" "$record"
  echo "[prevalidation] $(date '+%F %T') $fam $ds fold $fold: exit $rc after ${wall} s"
}

for task in "${TASKS[@]}"; do
  IFS=$'\t' read -r fam ds fold prof <<< "$task"
  record="$OUT_ROOT/$fam/${ds}_fold${fold}_process.json"
  if [[ "$RESUME" == "1" && -s "$record" ]] && grep -q '"exit": 0,' "$record"; then
    echo "[prevalidation] $fam $ds fold $fold: finished before -- RESUME=1 skip"
    continue
  fi
  # A task that did not finish starts over. The binary would continue its cut search from the
  # GA checkpoint and truncate the events log at the restart, so fold_begin..fold_end -- the
  # training cost the report divides -- would time only the part after it: a shorter fold,
  # silently. Its files go aside, not away.
  if [[ "$RESUME" == "1" ]]; then
    shopt -s nullglob
    leftovers=("$OUT_ROOT/$fam/"*"${ds}_fold${fold}"[._]*)
    shopt -u nullglob
    if ((${#leftovers[@]})); then
      aside="$OUT_ROOT/$fam/interrupted_$(date +%Y%m%d_%H%M%S)_${ds}_fold${fold}"
      mkdir -p "$aside"
      mv -- "${leftovers[@]}" "$aside/"
      echo "[prevalidation] $fam $ds fold $fold: did not finish before; ${#leftovers[@]} files moved to $aside, rerun from the start"
    fi
  fi
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
  echo "[prevalidation] $(date '+%F %T') start $fam $ds fold $fold"
  run_task "$fam" "$ds" "$fold" "$prof" &
done
wait || true

failed=()
for task in "${TASKS[@]}"; do
  IFS=$'\t' read -r fam ds fold _ <<< "$task"
  grep -q '"exit": 0,' "$OUT_ROOT/$fam/${ds}_fold${fold}_process.json" 2>/dev/null \
    || failed+=("$fam $ds fold $fold")
done
if ((${#failed[@]})); then
  printf '[prevalidation] FAILED: %s\n' "${failed[@]}" >&2
  echo "[prevalidation] see each task's .log; rerun with RESUME=1." >&2
  exit 1
fi

if [[ "${SKIP_POSTPROCESS:-0}" == "1" ]]; then
  echo "[prevalidation] SKIP_POSTPROCESS=1: the eight tasks finished; run 03_ and the report" \
       "where the paper is (.wiki/Experiments/Meeting01.md, 'Running it')"
  exit 0
fi

echo "[prevalidation] PCA / mean-frame references on the snn-ae folds"
"$PY" "$SCRIPTS/03_meeting01_pca_mean_baselines.py" --results-dir "$OUT_ROOT/snn-ae" \
  --run-tag "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"]["run_tag"])' "$PROFILE")" \
  --profile "$OUT_ROOT/snn-ae/profile.json"
"$PY" "$SCRIPTS/00_meeting01_prevalidation_report.py" --report --profile "$PROFILE" \
  --out-root "$OUT_ROOT" --data-dir "$PAPER_DATA"
echo "[prevalidation] done"
