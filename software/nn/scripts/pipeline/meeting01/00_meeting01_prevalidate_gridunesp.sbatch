#!/bin/bash
# 00_meeting01_prevalidate_gridunesp.sbatch — runs the baseline pre-validation
# (00_meeting01_prevalidate_baselines.sh: 4 families x 2 development folds = 8 one-thread
# processes) as ONE GridUnesp job under job-nanny. The stop handler, the staging and job-nanny's
# use are those of 01_meeting01_run_loso_gridunesp.sbatch, for the reasons its comments and the
# wiki's "job-nanny" and "Stopping" sections give (measured on this cluster, 2026-10-13); read
# them there, they are not repeated here.
#
# Why ONE job on ONE node, not an array. The pre-validation's training cost is a RATIO: each
# family's fold time over the SNN-AE's on the same fold. Eight array tasks would land on
# different nodes beside different co-tenants, so each family would be timed under its own
# conditions; one node runs all eight side by side under the same ones.
# Why 16 CPUs for 8 single-thread processes: a node's CPUs are hyperthreads (28 cores x 2), so 8
# CPUs can be 4 cores, two processes on each -- a slowdown that would fall on some families and
# not others, inside the ratio. 16 gives each process a core of its own.
#
# Usage (from ~/software/nn on GridUnesp, after gridunesp_deploy.sh):
#   sbatch scripts/pipeline/meeting01/00_meeting01_prevalidate_gridunesp.sbatch
#   RESUME=1 sbatch ...                      # keep finished tasks; unfinished ones start over
#   scancel --batch --signal=USR1 <jobid>    # stop early, results copied back
# Then, on the machine that has the paper: rsync results/meeting01_prevalidation home, run 03_ on
# its snn-ae/ and 00_meeting01_prevalidation_report.py --report (.wiki/Experiments/Meeting01.md).
# Smoke test of this path (tiny profile; keep the meeting01-loso prefix in --job-name, the deploy
# guard looks for it; -t under 15 min needs its own --signal):
#   PREVALIDATION_PROFILE=$HOME/prevalidation_smoke/smoke_profile.json \
#   PREVALIDATION_DIR=results/meeting01_prevalidation_smoke \
#   sbatch -p short -t 00:20:00 --signal=B:USR1@150 --job-name=meeting01-loso-preval-smoke \
#     scripts/pipeline/meeting01/00_meeting01_prevalidate_gridunesp.sbatch
#
#SBATCH --job-name=meeting01-loso-preval
# medium (7 days): the estimate is under a day, but a task cut by the limit must be rerun from
# its start (a resumed fold is not one interval), so the limit is generous.
#SBATCH --partition=medium
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=meeting01_loso_preval_%j.out
#SBATCH --error=meeting01_loso_preval_%j.err

set -euo pipefail

# Stop handler: see 01_meeting01_run_loso_gridunesp.sbatch ("Slurm signals only this shell").
STOPPED=0
NANNY_PID=""
stop_run() {
  STOPPED=1
  [[ -n "$NANNY_PID" ]] || return 0
  echo "[gridunesp] $(date '+%F %T') stop signal: ending the run; job-nanny copies the results back" >&2
  local pgid="" _
  for _ in {1..30}; do
    if [[ -s "$RUN_PGID_FILE" ]]; then pgid="$(<"$RUN_PGID_FILE")"; break; fi
    sleep 1
  done
  if [[ -z "$pgid" ]]; then
    echo "[gridunesp] no run process group recorded in $RUN_PGID_FILE -- cannot stop the run" >&2
    return 0
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
}
trap stop_run USR1 TERM

module load gridunesp
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build

# One thread per process, eight processes: NOT the CPU count, unlike the grid's sbatch (there one
# process gets every CPU of its task). The program gets slower with every thread it is given.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JOBS=8

export EXPERIMENT_CONFIRMED=1
export SKIP_BUILD=1
export SKIP_POSTPROCESS=1
export MEETING01_BUILD=max-performance
export RESUME="${RESUME:-0}"

export SHARED_FS=true
export WAIT_CHECKPOINT="${WAIT_CHECKPOINT:-1800}"
export VERBOSE=1

PROFILE_SRC="${PREVALIDATION_PROFILE:-src/experiments/meeting01/profiles/meeting01-prevalidation.json}"
RES="${PREVALIDATION_DIR:-results/meeting01_prevalidation}"
[[ "$RES" == results/* ]] || { echo "REFUSED: PREVALIDATION_DIR must be under results/ (job-nanny copies results/ back), got $RES" >&2; exit 1; }
BUILD_DIR="out/build/${MEETING01_BUILD}"
for f in SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.built" \
         "${BUILD_DIR}/src/experiments/meeting01/meeting01" "$PROFILE_SRC"; do
  [[ -e "$f" ]] || {
    echo "REFUSED: $f is missing -- deploy and build first (gridunesp_deploy.sh), or fix" \
         "PREVALIDATION_PROFILE." >&2
    exit 1
  }
done
if [[ "$RESUME" != "1" && -d "$RES" && -n "$(ls -A "$RES")" ]]; then
  echo "REFUSED: fresh run (RESUME unset) but $PWD/$RES is not empty. Move it aside first" \
       "(mv $RES ${RES}_before_${SLURM_JOB_ID}), or continue it with RESUME=1." >&2
  exit 1
fi

STAGE=".job-input-${SLURM_JOB_ID}"
RUN_PGID_FILE="/tmp/meeting01-run-${SLURM_JOB_ID}.pgid"
trap 'rm -rf -- "$STAGE" "$RUN_PGID_FILE"' EXIT
PROFILE_STAGED="src/experiments/meeting01/profiles/$(basename "$PROFILE_SRC")"
mkdir -p "$STAGE/scripts/pipeline" "$STAGE/src/experiments/meeting01/profiles" \
         "$STAGE/${BUILD_DIR}/src/experiments/meeting01" "$STAGE/results"
cp -a SOURCE_REVISION "$STAGE/"
cp -a scripts/pipeline/meeting01 "$STAGE/scripts/pipeline/"
cp -a "$PROFILE_SRC" "$STAGE/$PROFILE_STAGED"
cp -a "${BUILD_DIR}/SOURCE_REVISION.built" "$STAGE/${BUILD_DIR}/"
cp -a "${BUILD_DIR}/src/experiments/meeting01/meeting01" "$STAGE/${BUILD_DIR}/src/experiments/meeting01/"
if [[ "$RESUME" == "1" && -d "$RES" ]]; then
  mkdir -p "$STAGE/$(dirname "$RES")"
  cp -a "$RES" "$STAGE/$(dirname "$RES")/"
fi
export PREVALIDATION_PROFILE="$PROFILE_STAGED"
export PREVALIDATION_DIR="$RES"

cat > "$STAGE/start.sh" <<'START'
#!/bin/bash
# start.sh STAGE_IN_SUBMIT_DIR PGID_FILE RUN_SCRIPT -- written by
# 00_meeting01_prevalidate_gridunesp.sbatch for job-nanny; not meant to be run by hand.
set -euo pipefail
[[ "$1" == */.job-input-"$SLURM_JOB_ID" ]] || { echo "start.sh: refusing to delete '$1'" >&2; exit 1; }
rm -rf -- "$1"
exec setsid -w bash -c 'echo "$$" > "$1"; exec bash "$2"' start "$2" "$3"
START

export INPUT="$STAGE"
export OUTPUT="$STAGE/results"

if [[ "$STOPPED" == "1" ]]; then
  echo "[gridunesp] $(date '+%F %T') stop signal during setup -- not starting the run" >&2
  exit 143
fi
job-nanny bash "$STAGE/start.sh" "$PWD/$STAGE" "$RUN_PGID_FILE" \
  "$STAGE/scripts/pipeline/meeting01/00_meeting01_prevalidate_baselines.sh" &
NANNY_PID=$!
RC=0
while :; do
  wait "$NANNY_PID" && RC=0 || RC=$?
  kill -0 "$NANNY_PID" 2>/dev/null || break
done

if [[ "$STOPPED" == "1" ]]; then
  echo "[gridunesp] STOPPED before the run finished (time limit or scancel; run exit code $RC)." \
       "Results so far are in $PWD/$RES. Continue with: RESUME=1 sbatch" \
       "scripts/pipeline/meeting01/00_meeting01_prevalidate_gridunesp.sbatch" \
       "(unfinished tasks start over)" >&2
fi
exit "$RC"
