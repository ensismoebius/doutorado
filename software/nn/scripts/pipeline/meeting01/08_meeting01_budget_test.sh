#!/usr/bin/env bash
# 08_meeting01_budget_test.sh -- the budget test: EEGMMIDB ANN-AE by hidden width (65/128/256) and
# the four SNN-AE winners with their ANN-AE controls at 60 epochs, snapshotted at 30 (design and
# readings: 08_meeting01_budget_test.py's docstring). Local, CPU, one thread per process.
#
#   36 trainings, at most 60 epochs each, on the pre-validation's saved fold-0 windows, with the
#   design study's twin (07_meeting01_design_study.py, imported unchanged)
#   then  08_meeting01_budget_test.py --report  ->  <OUT_ROOT>/budget_{trainings,contrasts}.csv,
#         budget_report.txt (includes the determinism check against results/meeting01_design),
#         and <PAPER_DATA>/paper_budget.tex (the numbers of the paper's budget paragraph)
#
# COST: from the design study's measured epochs (2026-11-09): SNN-AE 11-16 s, ANN-AE ~3 s per
#   epoch (more at width 256); about 5-6 CPU-hours, 30-45 min on 11 processes. Refuses to start
#   without EXPERIMENT_CONFIRMED=1.
#
# Usage (from software/nn):
#   EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/08_meeting01_budget_test.sh
#   JOBS=N  BUDGET_DIR (default results/meeting01_budget; refused unless empty)  RESUME=1
#   PAPER_DATA (where paper_budget.tex goes; default: the paper's data/)
set -euo pipefail
export LC_ALL=C
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
SCRIPTS="scripts/pipeline/meeting01"
OUT_ROOT="${BUDGET_DIR:-results/meeting01_budget}"
PAPER_DATA="${PAPER_DATA:-$ROOT_DIR/../../documentation/07-articlesProduced/meeting01/data}"
JOBS="${JOBS:-$(( $(nproc) > 2 ? $(nproc) - 1 : 1 ))}"
RESUME="${RESUME:-0}"

if [[ "${EXPERIMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "REFUSED: the budget test trains 36 networks (most of an hour). Re-invoke with EXPERIMENT_CONFIRMED=1." >&2
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
"$PY" "$SCRIPTS/08_meeting01_budget_test.py" --self-test
mkdir -p "$OUT_ROOT"
for s in 07_meeting01_design_study.py 08_meeting01_budget_test.py; do
  cp "$SCRIPTS/$s" "$OUT_ROOT/$s.used"
done
sha() { sha256sum "$1" | cut -c1-16; }
echo "$(date '+%F %T') ${MEETING01_GIT_COMMIT} JOBS=${JOBS} threads=1 RESUME=${RESUME}" \
     "07_sha256=$(sha "$OUT_ROOT/07_meeting01_design_study.py.used")" \
     "08_sha256=$(sha "$OUT_ROOT/08_meeting01_budget_test.py.used")" >> "$OUT_ROOT/source_revisions.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" "$SCRIPTS/08_meeting01_budget_test.py" --run --out-root "$OUT_ROOT" --jobs "$JOBS"
"$PY" "$SCRIPTS/08_meeting01_budget_test.py" --report --out-root "$OUT_ROOT" --data-dir "$PAPER_DATA"
echo "[budget] done"
