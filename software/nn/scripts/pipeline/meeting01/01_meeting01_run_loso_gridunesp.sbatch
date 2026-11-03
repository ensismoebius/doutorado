#!/bin/bash
# 01_meeting01_run_loso_gridunesp.sbatch — submits the nested-LOSO training loop
# (01_meeting01_run_loso.sh) to GridUnesp's Slurm queue under job-nanny, the cluster's mandatory
# data-staging wrapper. Read .wiki/Guides/GridUnesp-Deployment.md before the first submission: the
# binary must already be BUILT and stamped (gridunesp_deploy.sh does both); this script only runs it.
#
# Usage (from ~/software/nn on GridUnesp, after gridunesp_deploy.sh):
#   sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch            # fresh run
#   RESUME=1 sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch   # continue
#   scancel --batch --signal=USR1 <jobid>    # stop early, results copied back (as at the limit)
# One task per (dataset, fold) instead of one long job -- 24 one-CPU tasks, resubmittable per
# task (see "Array mode" below): use scripts/pipeline/meeting01/submit_loso_array.sh, which
# submits `--array=<unfinished tasks> -c 1` and refuses to double-submit.
# Smoke test of this whole path before a long queue wait (~6 min: the early stop at the time limit
# is part of what it tests; then move results/meeting01 aside -- a fresh run refuses a non-empty
# one). Keep the meeting01-loso prefix in any --job-name: gridunesp_deploy.sh looks for it. Any
# -t under 15 min needs its own --signal, as here: the @900 below would arrive at the start.
#   WAIT_CHECKPOINT=60 sbatch -p short -t 00:06:00 -c 4 --signal=B:USR1@150 \
#     --job-name=meeting01-loso-smoke scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
#
#SBATCH --job-name=meeting01-loso
#SBATCH --partition=long
#SBATCH --time=30-00:00:00
# USR1 to this shell (B:) 15 min before the time limit: the stop handler below ends the run and
# lets job-nanny copy the results back and delete its work dir. Why the limit itself cannot do
# that: the "Slurm signals only this shell" comment under `set -euo pipefail`.
#SBATCH --signal=B:USR1@900
# --cpus-per-task=1: this program gets slower with every thread it is given (batch size 1
# makes each multiply too small to split; the wiki's "Threads" section: 0.41 s per step at 1
# thread, 9.2 s at 20 on a 12-core desktop, and job 5259474 at 20 CPUs took ~2 days an epoch).
# Until 2026-11-02 this said 20, the user's choice from earlier timing tests made before that
# measurement. submit_loso_array.sh passes -c 1 anyway; this default matters for a single job
# submitted without it. Each node has 56 CPUs (28 cores x 2 threads) but the partition is
# shared and nearly full: requests of 28, 26 and 12 all sat in PENDING ("Nodes required for
# job are DOWN, DRAINED or reserved ..."), a queue-priority effect of the `long` tier, not of
# the CPU count. The run is restartable (RESUME=1). The deploy's build step uses its own
# count (GRIDUNESP_BUILD_CPUS).
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=meeting01_loso_%j.out
#SBATCH --error=meeting01_loso_%j.err

set -euo pipefail

# ── Slurm signals only this shell ─────────────────────────────────────────────────────────────
# At the time limit (or a scancel) Slurm sends SIGTERM and, KillWait = 120 s later, SIGKILL. With
# ProctrackType=proctrack/cgroup the SIGTERM goes to the step's own task -- this shell -- and not
# to the processes under it; they only ever get the SIGKILL. Smoke job 5259424 (2026-10-13) shows
# it: its run kept training past the limit, job-nanny made one more periodic copy, and 120 s later
# everything died by signal 9 (sacct: batch step CANCELLED, 0:9) -- no final copy, WORK_DIR left
# on /store, and this shell's own trap never ran either, because a shell waiting on a FOREGROUND
# command defers every trap until that command returns. Hence job-nanny goes to the background
# (`wait` returns as soon as a trapped signal arrives) and this handler ends the run itself:
# job-nanny then sees its command return and does its normal finish -- final copy back, WORK_DIR
# deleted. USR1 comes 15 min ahead (--signal above); TERM is the limit itself or a plain scancel.
# Installed first, so that a signal during the staging below is not fatal (USR1's default action
# is to terminate); before job-nanny starts it only sets STOPPED, checked right before the launch.
STOPPED=0
NANNY_PID=""
stop_run() {
  STOPPED=1
  [[ -n "$NANNY_PID" ]] || return 0   # during setup: reported by the check before the launch
  echo "[gridunesp] $(date '+%F %T') stop signal: ending the run; job-nanny copies the results back" >&2
  local pgid="" _
  for _ in {1..30}; do   # job-nanny may still be copying the stage in; the run starts after that
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

# job-nanny is NOT on a batch job's PATH: it comes with the `gridunesp` module
# (/opt/gridunesp/dist/gridunesp/bin/job-nanny -> goo-job-nanny-v2), which only a LOGIN shell
# loads automatically (GridUnesp docs v3, "Melhorando o Script de Submissão"). Without this line
# the first submission (job 5245836) waited five days in the queue and then died in one second:
# "job-nanny: command not found".
module load gridunesp
module load miniconda/24.4.0-libmamba
# `module load` only puts conda on PATH -- `conda activate` needs the shell hook that `conda init`
# normally installs into an interactive login shell's rc file, which a batch job never sources.
eval "$(conda shell.bash hook)"
conda activate meeting01-build

# The docs' OpenMP example does this; it also stops OpenMP from defaulting to every CPU of the
# node should the allocation not be enforced as an affinity mask.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# OpenBLAS sizes its own thread pool separately from OpenMP; pin it to the same count so the two
# cannot multiply (job 5259474, 20 CPUs, ran ~1000x slower than expected: see the wiki's
# "Threads" section).
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export EXPERIMENT_CONFIRMED=1
export SKIP_BUILD=1
export SKIP_POSTPROCESS=1
export MEETING01_BUILD=max-performance
export RESUME="${RESUME:-0}"

# ── Array mode: one task per (dataset, outer fold) ────────────────────────────────────────────
# `sbatch --array=0-23 -c 1 ...` (submit_loso_array.sh builds that line, and resubmits only the
# unfinished tasks). Task t is dataset ARRAY_DATASETS[t / 6], fold t % 6 -- the profile's four
# datasets and cv_num_folds = 6, which the check below ties to the profile so a change there
# cannot silently shift the mapping. Each task is its own Slurm job (own %j, own job-nanny work
# dir on /store), runs ONE fold in ONE process, and RESUME=1 on a resubmitted task reruns that fold,
# continuing only the search that was cut mid-way from its stamped GA checkpoint. The point: the measured cost of this program is per-thread (see the
# wiki's "Threads" section), so 24 one-CPU tasks beat one 20-CPU job running the folds in series.
ARRAY_DATASETS=(fsdd audiomnist eegmmidb siena)
ARRAY_FOLDS=6
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  PROFILE_JSON="src/experiments/meeting01/profiles/meeting01-loso.json"
  python3 - "$PROFILE_JSON" "$ARRAY_FOLDS" "${ARRAY_DATASETS[*]}" <<'PY' || exit 1
import json, sys
profile, folds, datasets = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
p = json.load(open(profile))
if p["dataset"]["cv_num_folds"] != folds or p["evaluation"]["datasets"] != datasets:
    sys.exit(f"REFUSED: the array mapping assumes {datasets} x {folds} folds, but {profile} has "
             f"{p['evaluation']['datasets']} x {p['dataset']['cv_num_folds']}. Edit ARRAY_DATASETS / "
             "ARRAY_FOLDS in the sbatch to match.")
PY
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#ARRAY_DATASETS[@]} * ARRAY_FOLDS )); then
    echo "REFUSED: array task ${SLURM_ARRAY_TASK_ID} is outside 0..$(( ${#ARRAY_DATASETS[@]} * ARRAY_FOLDS - 1 ))" >&2
    exit 1
  fi
  export DATASETS="${ARRAY_DATASETS[$(( SLURM_ARRAY_TASK_ID / ARRAY_FOLDS ))]}"
  export FOLDS="$(( SLURM_ARRAY_TASK_ID % ARRAY_FOLDS ))"
  export PROVENANCE_LOG_SUFFIX="_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "[gridunesp] array task ${SLURM_ARRAY_TASK_ID}: dataset ${DATASETS}, fold ${FOLDS}"
fi

# ── job-nanny, as its source (goo-job-nanny-v2, read 2026-10-13) actually behaves ──────────────
#   - INPUT and OUTPUT are required ("INPUT is missing ..." and exit 3 otherwise);
#   - `rsync -a -u $INPUT $WORK_DIR`, WORK_DIR=/store/$USER/$SLURM_JOB_ID/ under SHARED_FS=true,
#     WITHOUT -R: a nested path arrives flattened to its basename, so INPUT is ONE directory that
#     already has the layout the run expects (built below);
#   - it cd's into WORK_DIR, runs the command, and every WAIT_CHECKPOINT seconds copies OUTPUT
#     back to the submit directory (rsync -a -u, nothing deleted there);
#   - when the command ends it copies OUTPUT back, deletes WORK_DIR and exits with the command's
#     own code -- only then: it has no signal handling, so a job-nanny that is SIGKILLed (what a
#     time limit or scancel does to it, see the end) leaves WORK_DIR on /store with the newest
#     state and the submit directory with the last periodic copy.
# /store rather than node-local /tmp: no automatic cleanup and it survives a node failure (docs
# v3, storage guide), while /tmp is wiped by a cron for users with no job on that node.
export SHARED_FS=true
export WAIT_CHECKPOINT="${WAIT_CHECKPOINT:-1800}"   # results reach ~/software/nn every 30 min
export VERBOSE=1   # job-nanny's own log lines (its flag is inverted: this also quiets its rsync)

RES=results/meeting01
BUILD_DIR="out/build/${MEETING01_BUILD}"
for f in SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.built" \
         "${BUILD_DIR}/src/experiments/meeting01/meeting01"; do
  [[ -e "$f" ]] || {
    echo "REFUSED: $PWD/$f is missing -- this tree is not deployed and built. Run" \
         "scripts/pipeline/meeting01/gridunesp_deploy.sh on your machine first." >&2
    exit 1
  }
done

# A fresh run must start from nothing. Before job-nanny the run script cleared the old
# checkpoints/models in place; now it clears them inside WORK_DIR, and the copy back (which never
# deletes) would leave the old ones in ~/software/nn next to the new rows.
if [[ "$RESUME" != "1" && -z "${SLURM_ARRAY_TASK_ID:-}" && -d "$RES" && -n "$(ls -A "$RES")" ]]; then
  echo "REFUSED: fresh run (RESUME unset) but $PWD/$RES is not empty. Move it aside first" \
       "(mv $RES ${RES}_before_${SLURM_JOB_ID}), or continue it with RESUME=1." >&2
  exit 1
fi
# In an array the other tasks copy their results into the same $RES every WAIT_CHECKPOINT seconds,
# so "$RES is empty" is false for a task that merely queued longer. A task's own fold is what must
# not exist: every file of a fold starts meeting01_loso_<dataset>_fold<f>_.
if [[ "$RESUME" != "1" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] &&
   compgen -G "$RES/meeting01_loso_${DATASETS}_fold${FOLDS}_*" >/dev/null; then
  echo "REFUSED: fresh task for ${DATASETS} fold ${FOLDS} but $PWD/$RES already holds its files." \
       "Continue it with RESUME=1, or move those files aside" \
       "(mv $RES/meeting01_loso_${DATASETS}_fold${FOLDS}_* elsewhere)." >&2
  exit 1
fi

# INPUT: exactly what the run reads relative to its own root, in the same layout. Datasets are
# read from their absolute /home paths (the profile's dataset.sources[].root), once per fold, and
# every fold's process loads libstatistics.so and libmatio* from the build tree on /home (the
# binary's RPATH; only the executable is staged) -- which is why gridunesp_deploy.sh refuses to
# run while a meeting01-loso job does. (latex_data_dir, which used to receive a few small files
# per fold on /home, is empty since 2026-11-02: nothing read them.)
STAGE=".job-input-${SLURM_JOB_ID}"
RUN_PGID_FILE="/tmp/meeting01-run-${SLURM_JOB_ID}.pgid"
# Covers a job that fails before job-nanny runs start.sh; after that, start.sh has already
# deleted the stage (below) and this finds nothing.
trap 'rm -rf -- "$STAGE" "$RUN_PGID_FILE"' EXIT
mkdir -p "$STAGE/scripts/pipeline" "$STAGE/src/experiments/meeting01/profiles" \
         "$STAGE/${BUILD_DIR}/src/experiments/meeting01" "$STAGE/results"
cp -a SOURCE_REVISION "$STAGE/"
cp -a scripts/pipeline/meeting01 "$STAGE/scripts/pipeline/"
cp -a src/experiments/meeting01/profiles/meeting01-loso.json "$STAGE/src/experiments/meeting01/profiles/"
cp -a "${BUILD_DIR}/SOURCE_REVISION.built" "$STAGE/${BUILD_DIR}/"
cp -a "${BUILD_DIR}/src/experiments/meeting01/meeting01" "$STAGE/${BUILD_DIR}/src/experiments/meeting01/"
if [[ "$RESUME" == "1" && -d "$RES" ]]; then
  cp -a "$RES" "$STAGE/results/"
fi

# What job-nanny runs, from WORK_DIR, once it has copied the stage there. It deletes the stage
# in ~/software/nn at once (from then on only the WORK_DIR copy is used, and a job that gets
# killed never reaches the EXIT trap above -- smoke job 5259424 left its stage behind), then
# starts the run as the leader of a new process group and writes the group id down, so that
# stop_run can end the run, and only the run, while job-nanny goes on to its final copy.
cat > "$STAGE/start.sh" <<'START'
#!/bin/bash
# start.sh STAGE_IN_SUBMIT_DIR PGID_FILE RUN_SCRIPT -- written by
# 01_meeting01_run_loso_gridunesp.sbatch for job-nanny; not meant to be run by hand.
set -euo pipefail
[[ "$1" == */.job-input-"$SLURM_JOB_ID" ]] || { echo "start.sh: refusing to delete '$1'" >&2; exit 1; }
rm -rf -- "$1"
exec setsid -w bash -c 'echo "$$" > "$1"; exec bash "$2"' start "$2" "$3"
START

# OUTPUT is relative to WORK_DIR; rsync copies the directory `results` into the submit
# directory, i.e. onto ~/software/nn/results.
export INPUT="$STAGE"
export OUTPUT="$STAGE/results"

# The stop handler (installed at the top) found a signal before this point: do not start a run
# that would only be SIGKILLed 120 s later.
if [[ "$STOPPED" == "1" ]]; then
  echo "[gridunesp] $(date '+%F %T') stop signal during setup -- not starting the run" >&2
  exit 143
fi
job-nanny bash "$STAGE/start.sh" "$PWD/$STAGE" "$RUN_PGID_FILE" \
  "$STAGE/scripts/pipeline/meeting01/01_meeting01_run_loso.sh" &
NANNY_PID=$!
RC=0
while :; do
  wait "$NANNY_PID" && RC=0 || RC=$?
  # A trap ends `wait` early (status > 128) while job-nanny is still running -- wait again.
  kill -0 "$NANNY_PID" 2>/dev/null || break
done

if [[ "$STOPPED" == "1" ]]; then
  echo "[gridunesp] STOPPED before the run finished (time limit or scancel; run exit code $RC)." \
       "Results so far are in $PWD/$RES. Continue with:" \
       "$( [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] \
            && echo 'scripts/pipeline/meeting01/submit_loso_array.sh   (resubmits only the unfinished tasks)' \
            || echo 'RESUME=1 sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch' )" >&2
fi
exit "$RC"
