#!/usr/bin/env bash
# 07_meeting01_design_study.sh -- the design study: the logged PyTorch twin of the SNN-AE, crossed
# readout x decoder and encoder neuron x initializer, at the four SNN-AE winners of the
# results_format 11 pre-validation, 4 seeds each (design and rationale: 07_meeting01_design_study.py's
# docstring). Local, CPU, one thread per process.
#
#   208 trainings (112 spiking) on the pre-validation's saved fold-0 windows
#   then  07_meeting01_design_study.py --report  ->  <OUT_ROOT>/design_{trainings,cells,contrasts}.csv,
#         <paper>/data/paper_design.tex, paper_design_shares.tex, paper_design_stats.tex
#
# COST: measured 2026-11-09 on the 12-core desktop under load 14: 18-41 s per epoch for a spiking
#   training (T = 19-31, batch 1), ~6 s for an analog one; at most 30 epochs. About 19 CPU-hours,
#   2-3 hours on 10 processes. Refuses to start without EXPERIMENT_CONFIRMED=1.
#
# Usage (from software/nn):
#   EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/07_meeting01_design_study.sh
#   JOBS=N           processes at a time (default nproc - 2)
#   DESIGN_DIR       output dir (default results/meeting01_design); refused unless empty
#   RESUME=1         continue a stopped run in DESIGN_DIR: trainings with a row are kept only if
#                    that row's source revision is this one (otherwise refused)
#   PREVALIDATION    the pre-validation results (default results/meeting01_prevalidation)
#   PAPER_DATA       where the paper_design*.tex go (default: the paper's data/)
set -euo pipefail
export LC_ALL=C
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
SCRIPTS="scripts/pipeline/meeting01"
OUT_ROOT="${DESIGN_DIR:-results/meeting01_design}"
PRE="${PREVALIDATION:-results/meeting01_prevalidation}"
PAPER_DATA="${PAPER_DATA:-/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/meeting01/data}"
JOBS="${JOBS:-$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))}"
RESUME="${RESUME:-0}"

if [[ "${EXPERIMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "REFUSED: the design study trains 208 networks (hours). Re-invoke with EXPERIMENT_CONFIRMED=1." >&2
  exit 1
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "REFUSED: JOBS must be a positive integer (got '$JOBS')" >&2; exit 1; }
if [[ "$RESUME" != "1" && -d "$OUT_ROOT" && -n "$(ls -A "$OUT_ROOT")" ]]; then
  echo "REFUSED: $OUT_ROOT is not empty. Move it aside first, or continue it with RESUME=1." >&2
  exit 1
fi
PY="$ROOT_DIR/.venv/bin/python3"
[[ -x "$PY" ]] || { echo "REFUSED: $PY is missing (the twin needs the venv's torch)" >&2; exit 1; }
# shellcheck source=./_provenance.sh
source "$ROOT_DIR/$SCRIPTS/_provenance.sh"
provenance_resolve "$ROOT_DIR" "$PY"

"$PY" "$SCRIPTS/07_meeting01_design_study.py" --self-test
mkdir -p "$OUT_ROOT"
# The exact script that trains, kept beside its results: a "+dirty" revision label says the tree
# had changes, this copy and its hash say which.
cp "$SCRIPTS/07_meeting01_design_study.py" "$OUT_ROOT/07_meeting01_design_study.py.used"
used_sha="$(sha256sum "$OUT_ROOT/07_meeting01_design_study.py.used" | cut -c1-16)"
echo "$(date '+%F %T') ${MEETING01_GIT_COMMIT} JOBS=${JOBS} threads=1 RESUME=${RESUME} script_sha256=${used_sha} prevalidation=${PRE}" \
  >> "$OUT_ROOT/source_revisions.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" "$SCRIPTS/07_meeting01_design_study.py" --run --prevalidation "$PRE" --out-root "$OUT_ROOT" --jobs "$JOBS"
"$PY" "$SCRIPTS/07_meeting01_design_study.py" --report --prevalidation "$PRE" --out-root "$OUT_ROOT" \
  --data-dir "$PAPER_DATA" | tee "$OUT_ROOT/design_contrasts.txt"
echo "[design] done"
