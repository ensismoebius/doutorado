#!/bin/bash
# thesis_gridunesp.sbatch -- one phase of the thesis experiment (Experiment05) on GridUnesp, as a
# job array under job-nanny. The binary must already be BUILT and stamped on the cluster:
# scripts/pipeline/meeting01/gridunesp_deploy.sh does both (its default targets include thesis)
# and sends ~/database.sqlite, the profiles' dataset.root.
#
# Usage (from ~/software/nn on GridUnesp, after the deploy):
#   THESIS_PHASE=phase00 sbatch --array=0-15 scripts/pipeline/thesis/thesis_gridunesp.sbatch
#   THESIS_PHASE=phase01 sbatch --array=0-31 -p medium -t 5-00:00:00 scripts/pipeline/thesis/thesis_gridunesp.sbatch
#   scancel --batch --signal=USR1 <jobid>_<task>   # stop a task early; its results come home
#
# ── Why an array of BATCHES of profiles ─────────────────────────────────────────────────────
# Phase 00 is 214 profiles that each take seconds to ~15 minutes; Phase 01 is 32 that take hours.
# Every profile first loads the decoded dataset (~/database.sqlite.e05dscache, 1.6 GB) from /home.
# One Slurm task per Phase 00 profile would read it 214 times from /home, the heavy /home I/O the
# job-nanny page warns about; a task that runs several profiles in a row reads it once and serves
# the rest from that node's page cache. So task t of an N-task array runs the profiles whose
# position in the sorted list is t, t+N, t+2N, ... (interleaved, so the 24 slow autoencoder
# profiles of Phase 00 spread over the tasks instead of piling into one):
#
#   sorted phase00 list   p00_ae_ann_base_eeg  p00_ae_ann_base_voice  ...  p00_pca_k8_voice
#   position              0                    1                      ...  213
#   task (N = 16)         0                    1                      ...  213 % 16 = 5
#
# The array must be 0..N-1 with nothing left out (checked): the mapping is a function of N, so a
# partial resubmission (--array=3,7) would hand task 3 another task's profiles. Resubmit the whole
# array instead -- a task whose profiles are all done exits in seconds.
#
# ── Done markers, and why a marker carries the code revision ─────────────────────────────────
# A profile is done when the binary exited 0 AND every summary its repeats should write exists;
# then results/thesis/<phase>/.done/<profile>.done is written, holding the deploy's revision
# label (commit/tree hash, see scripts/pipeline/meeting01/_provenance.sh). A resubmitted task
# skips its profiles that are done. A marker of ANOTHER revision anywhere in that directory
# refuses the task: the phase would otherwise mix results of two binaries -- what happened to the
# July run, whose Phase 01 numbers came from code later found defective -- and nothing
# downstream could tell them apart. Move the old directory aside to run a new revision.
#
# ── Failures are not skipped ─────────────────────────────────────────────────────────────────
# A profile that fails (non-zero exit, or a summary missing afterwards) stops its task at once:
# the task exits non-zero, results/thesis/<phase>/.failed/<profile>.failed holds the exit code and
# the log path, and the remaining profiles of that task do not run. Nothing is retried, nothing
# is passed over. Fix the cause, redeploy if the code changed (then move the phase aside: see
# above), and resubmit the array.
#
# After all tasks are done, bring the phase home and post-process it locally (the .wiki page
# Experiments/Thesis.md, "Running on GridUnesp").
#
#SBATCH --job-name=thesis-run
#SBATCH --partition=short
#SBATCH --time=1-00:00:00
# USR1 to this shell 15 min before the limit, for the same reason as the meeting01 sbatch: Slurm
# signals only this shell, and its children only get the SIGKILL (see that file's comment).
#SBATCH --signal=B:USR1@900
# 4 CPUs: the handcrafted extractor is OpenMP-parallel over samples; measured locally
# (2026-11-12) one autoencoder epoch takes 3-6 s at 4 threads, the whole of Phase 00 ~8 h serial.
#SBATCH --cpus-per-task=4
# The decoded dataset alone is 1.6 GB in memory; the voice SNN-AE profiles peaked at ~4.4 GB RSS
# locally. Out of memory is a loud failure (the task is killed and has no marker).
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=thesis_run_%A_%a.out
#SBATCH --error=thesis_run_%A_%a.err

set -euo pipefail

# ── Stop handler (same mechanism as 01_meeting01_run_loso_gridunesp.sbatch) ─────────────────────
STOPPED=0
NANNY_PID=""
stop_run() {
  STOPPED=1
  [[ -n "$NANNY_PID" ]] || return 0
  echo "[thesis-grid] $(date '+%F %T') stop signal: ending the run; job-nanny copies the results back" >&2
  local pgid="" _
  for _ in {1..30}; do
    if [[ -s "$RUN_PGID_FILE" ]]; then pgid="$(<"$RUN_PGID_FILE")"; break; fi
    sleep 1
  done
  if [[ -z "$pgid" ]]; then
    echo "[thesis-grid] no run process group recorded in $RUN_PGID_FILE -- cannot stop the run" >&2
    return 0
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
}
trap stop_run USR1 TERM

refuse() { echo "REFUSED: $*" >&2; exit 1; }

module load gridunesp
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ── What this task runs ───────────────────────────────────────────────────────────────────────
PHASE="${THESIS_PHASE:-}"
case "$PHASE" in
  phase00|phase01) ;;
  *) refuse "THESIS_PHASE must be phase00 or phase01 (got '${PHASE}'): THESIS_PHASE=phase00 sbatch --array=0-15 $0" ;;
esac
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] ||
  refuse "submit as an array (sbatch --array=0-<N-1> ...); see the header of this file"
(( SLURM_ARRAY_TASK_MIN == 0 && SLURM_ARRAY_TASK_COUNT == SLURM_ARRAY_TASK_MAX + 1 )) ||
  refuse "the array must be 0..N-1 with no gaps (got min ${SLURM_ARRAY_TASK_MIN}, max" \
         "${SLURM_ARRAY_TASK_MAX}, count ${SLURM_ARRAY_TASK_COUNT}): the profile-to-task mapping" \
         "depends on N. Resubmit the whole array; finished tasks exit in seconds."
NTASKS="$SLURM_ARRAY_TASK_COUNT"
TASK="$SLURM_ARRAY_TASK_ID"

BUILD_DIR="out/build/max-performance"
BIN="${BUILD_DIR}/src/experiments/thesis/thesis"
for f in SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.thesis.built" "$BIN"; do
  [[ -e "$f" ]] || refuse "$PWD/$f is missing -- this tree is not deployed and built for thesis." \
                          "Run scripts/pipeline/meeting01/gridunesp_deploy.sh on your machine first."
done
cmp -s SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.thesis.built" ||
  refuse "the thesis binary was not built from the sources now on disk (SOURCE_REVISION differs" \
         "from ${BUILD_DIR}/SOURCE_REVISION.thesis.built): a deploy synced and then did not build" \
         "it. Rerun gridunesp_deploy.sh."
source scripts/pipeline/meeting01/_provenance.sh
REVISION="$(provenance_label SOURCE_REVISION)"

PROFILE_DIR="src/experiments/thesis/profiles/${PHASE}"
mapfile -t ALL_PROFILES < <(find "$PROFILE_DIR" -maxdepth 1 -name '*.json' | sort)
(( ${#ALL_PROFILES[@]} > 0 )) || refuse "no profiles in $PWD/$PROFILE_DIR"
(( NTASKS <= ${#ALL_PROFILES[@]} )) ||
  refuse "$NTASKS tasks for ${#ALL_PROFILES[@]} profiles: some tasks would have nothing to run"

RES="results/thesis/${PHASE}"
DONE_DIR="${RES}/.done"
FAILED_DIR="${RES}/.failed"

# A marker of another revision anywhere in the phase: refuse (see the header).
if [[ -d "$DONE_DIR" ]]; then
  while IFS= read -r -d '' marker; do
    if [[ "$(<"$marker")" != "$REVISION" ]]; then
      refuse "$PWD/$marker was made by revision $(<"$marker"); this deploy is $REVISION. One" \
             "phase directory must not mix two binaries' results: move $PWD/$RES aside" \
             "(mv $RES ${RES}_$(date +%Y%m%d)) and resubmit the whole array."
    fi
  done < <(find "$DONE_DIR" -type f -name '*.done' -print0)   # -type f: ".done" itself matches
fi

MINE=()
for (( i = TASK; i < ${#ALL_PROFILES[@]}; i += NTASKS )); do
  stem="$(basename "${ALL_PROFILES[$i]}" .json)"
  [[ -f "${DONE_DIR}/${stem}.done" ]] && continue
  MINE+=("${ALL_PROFILES[$i]}")
done
echo "[thesis-grid] ${PHASE} task ${TASK}/${NTASKS}, revision ${REVISION}:" \
     "${#MINE[@]} profile(s) to run"
(( ${#MINE[@]} > 0 )) || { echo "[thesis-grid] every profile of this task is done"; exit 0; }
# A failure marker of one of MY profiles is from an earlier attempt of this task: this attempt
# reruns that profile, so it no longer describes the state.
for p in "${MINE[@]}"; do rm -f "${FAILED_DIR}/$(basename "$p" .json).failed"; done

# ── Stage (INPUT for job-nanny): exactly what run.sh reads, in the same layout ────────────────
# The dataset is NOT staged: the profiles read ~/database.sqlite (+ its cache) from /home, once
# per profile, from the node's page cache after the first (see the header). The binary loads this
# project's shared libraries from the build tree on /home (RPATH), which is why the deploy refuses
# while a thesis-run job is RUNNING.
STAGE=".job-input-${SLURM_JOB_ID}"
RUN_PGID_FILE="/tmp/thesis-run-${SLURM_JOB_ID}.pgid"
trap 'rm -rf -- "$STAGE" "$RUN_PGID_FILE"' EXIT
mkdir -p "$STAGE/${PROFILE_DIR}" "$STAGE/${BUILD_DIR}/src/experiments/thesis" \
         "$STAGE/${DONE_DIR}" "$STAGE/${FAILED_DIR}" "$STAGE/${RES}/logs"
cp -a SOURCE_REVISION "$STAGE/"
cp -a "$BIN" "$STAGE/${BIN}"
for p in "${MINE[@]}"; do cp -a "$p" "$STAGE/${PROFILE_DIR}/"; done
printf '%s\n' "${MINE[@]}" > "$STAGE/profiles.txt"
printf '%s\n' "$REVISION" > "$STAGE/revision.txt"

# run.sh: runs this task's profiles in order, from WORK_DIR, stopping at the first failure.
cat > "$STAGE/run.sh" <<'RUN'
#!/bin/bash
# run.sh -- written by thesis_gridunesp.sbatch for job-nanny; not meant to be run by hand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BIN="out/build/max-performance/src/experiments/thesis/thesis"
REVISION="$(<revision.txt)"
while IFS= read -r profile; do
  stem="$(basename "$profile" .json)"
  res="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["results_dir"])' "$profile")"
  log="${res}/logs/${stem}.log"
  mkdir -p "${res}/logs" "${res}/.done" "${res}/.failed"
  echo "[thesis-grid] $(date '+%F %T') start ${stem}"
  start=$SECONDS
  rc=0
  "$BIN" --config "$profile" > "$log" 2>&1 || rc=$?
  # Every file the repeats always write (thesis.cpp: summary and paraconsistent CSV, whatever
  # the phase): e05_<run_tag>_<kind> for one repeat, e05_<run_tag>_rep<k>_<kind> for k < repeats
  # otherwise -- the writers add "e05_" to a run_tag that already starts with it, the same rule
  # 01_thesis_phase00_rank.py's expected_csvs() builds its names from.
  missing="$(python3 - "$profile" "$res" <<'PY'
import json, os, sys
p = json.load(open(sys.argv[1]))
tag, reps = p["experiment"]["run_tag"], p["experiment"]["repeats"]
tags = [tag] if reps == 1 else [f"{tag}_rep{k}" for k in range(reps)]
names = [f"e05_{t}_{kind}" for t in tags for kind in ("summary.json", "paraconsistent.csv")]
print(" ".join(n for n in names if not os.path.isfile(os.path.join(sys.argv[2], n))))
PY
)"
  if [[ "$rc" != 0 || -n "$missing" ]]; then
    printf 'exit=%s\nmissing=%s\nlog=%s\nrevision=%s\n' "$rc" "$missing" "$log" "$REVISION" \
      > "${res}/.failed/${stem}.failed"
    echo "[thesis-grid] FAILED ${stem}: exit ${rc}${missing:+, missing ${missing}}; last lines of ${log}:" >&2
    tail -c 4000 "$log" | tr '\r' '\n' | tail -n 30 >&2
    exit 1
  fi
  printf '%s\n' "$REVISION" > "${res}/.done/${stem}.done"
  echo "[thesis-grid] $(date '+%F %T') done ${stem} ($(( SECONDS - start )) s)"
done < profiles.txt
RUN

# start.sh: deletes the stage in the submit directory (WORK_DIR has its copy now) and runs run.sh
# as the leader of its own process group, whose id stop_run needs (as in the meeting01 sbatch).
cat > "$STAGE/start.sh" <<'START'
#!/bin/bash
set -euo pipefail
[[ "$1" == */.job-input-"$SLURM_JOB_ID" ]] || { echo "start.sh: refusing to delete '$1'" >&2; exit 1; }
rm -rf -- "$1"
setsid -w bash -c 'echo "$$" > "$1"; exec bash "$2"' start "$2" "$3"
START

export SHARED_FS=true
export WAIT_CHECKPOINT="${WAIT_CHECKPOINT:-900}"   # results reach ~/software/nn every 15 min
export VERBOSE=1
export INPUT="$STAGE"
export OUTPUT="$STAGE/results"

if [[ "$STOPPED" == "1" ]]; then
  echo "[thesis-grid] $(date '+%F %T') stop signal during setup -- not starting the run" >&2
  exit 143
fi
job-nanny bash "$STAGE/start.sh" "$PWD/$STAGE" "$RUN_PGID_FILE" "$STAGE/run.sh" &
NANNY_PID=$!
RC=0
while :; do
  wait "$NANNY_PID" && RC=0 || RC=$?
  kill -0 "$NANNY_PID" 2>/dev/null || break
done

if [[ "$STOPPED" == "1" ]]; then
  echo "[thesis-grid] STOPPED before the task finished (time limit or scancel; exit $RC). Profiles" \
       "without a .done marker rerun on resubmission of the whole array." >&2
fi
exit "$RC"
