#!/usr/bin/env bash
# 05_meeting01_t1_transformer.sh -- T1: the Transformer-AE winners of the pre-validation retrained
# with a learning-rate warm-up and with pre-norm blocks (design and rationale:
# 05_meeting01_t1_transformer.py's docstring). Local, CPU, one compute thread per process.
#
#   12 processes = 2 development folds (fsdd, eegmmidb) x 2 winners x 3 arms (A control,
#   B warm-up, C pre-norm), each 4 trainings (a GA evaluation and a final fit per seed 42, 43)
#   then  05_meeting01_t1_transformer.py --report  ->  <OUT_ROOT>/t1_summary.csv,
#         <paper>/data/paper_t1.tex
#
# COST: measured 2026-11-04 (FSDD only, 6 processes at once, 12-core desktop): 5-15 min per
#   process; EEG winners are wider (d_model 79-96). Refuses to start without
#   EXPERIMENT_CONFIRMED=1.
#
# Usage (from software/nn):
#   EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/05_meeting01_t1_transformer.sh
#   JOBS=N        processes at a time (default 12, all of them)
#   SKIP_BUILD=1  reuse the binary (it must be built from these sources)
#   T1_DIR        output dir (default results/meeting01_t1); refused unless empty
#   PAPER_DATA    where paper_t1.tex goes (default: the paper's data/)
set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BUILD="max-performance"
BIN="out/build/${BUILD}/src/experiments/meeting01/meeting01"
SCRIPTS="scripts/pipeline/meeting01"
BASE="src/experiments/meeting01/profiles/meeting01-prevalidation.json"
OUT_ROOT="${T1_DIR:-results/meeting01_t1}"
PAPER_DATA="${PAPER_DATA:-/home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/meeting01/data}"
JOBS="${JOBS:-12}"

if [[ "${EXPERIMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "REFUSED: T1 trains 48 networks (tens of minutes). Re-invoke with EXPERIMENT_CONFIRMED=1." >&2
  exit 1
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "REFUSED: JOBS must be a positive integer (got '$JOBS')" >&2; exit 1; }
if [[ -d "$OUT_ROOT" && -n "$(ls -A "$OUT_ROOT")" ]]; then
  echo "REFUSED: $OUT_ROOT is not empty. Move it aside first." >&2
  exit 1
fi

PY="python3"
[[ -x "$ROOT_DIR/.venv/bin/python3" ]] && PY="$ROOT_DIR/.venv/bin/python3"
# shellcheck source=./_provenance.sh
source "$ROOT_DIR/$SCRIPTS/_provenance.sh"
provenance_resolve "$ROOT_DIR" "$PY"

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "[t1] SKIP_BUILD=1 but no binary at $BIN" >&2; exit 1; }
  provenance_check_binary_is_current "$ROOT_DIR" "$BUILD"
else
  cmake --preset="$BUILD"
  cmake --build --preset="$BUILD" -j"$(nproc)" --target meeting01
  provenance_mark_built "$ROOT_DIR" "$BUILD"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT_ROOT"
echo "$(date '+%F %T') ${MEETING01_GIT_COMMIT} JOBS=${JOBS} threads=1 base=${BASE}" >> "$OUT_ROOT/source_revisions.log"
mapfile -t TASKS < <("$PY" "$SCRIPTS/05_meeting01_t1_transformer.py" --profiles --base "$BASE" --out-root "$OUT_ROOT")
((${#TASKS[@]} > 0)) || { echo "[t1] no task" >&2; exit 1; }

run_task() {
  local name="$1" ds="$2" fold="$3" prof="$4" t0 rc=0 wall
  t0="$(date +%s.%N)"
  "$BIN" --comparative-config "$prof" --dataset "$ds" --cv-fold "$fold" --no-tui > "$OUT_ROOT/$name.log" 2>&1 || rc=$?
  wall="$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.3f", b - a }')"
  printf '{"name": "%s", "exit": %d, "wall_s": %s, "source_revision": "%s"}\n' \
    "$name" "$rc" "$wall" "$MEETING01_GIT_COMMIT" > "$OUT_ROOT/${name}_process.json.tmp"
  mv "$OUT_ROOT/${name}_process.json.tmp" "$OUT_ROOT/${name}_process.json"
  echo "[t1] $(date '+%F %T') $name: exit $rc after ${wall} s"
}

for task in "${TASKS[@]}"; do
  IFS=$'\t' read -r name ds fold prof <<< "$task"
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
  run_task "$name" "$ds" "$fold" "$prof" &
done
wait || true

"$PY" "$SCRIPTS/05_meeting01_t1_transformer.py" --report --out-root "$OUT_ROOT" --data-dir "$PAPER_DATA"
echo "[t1] done"
