#!/bin/bash
# thesis_gridunesp.sbatch -- one phase of the thesis experiment (Experiment05) on GridUnesp, as a
# job array under job-nanny. The binary must already be BUILT and stamped on the cluster:
# scripts/pipeline/meeting01/gridunesp_deploy.sh does both (its default targets include thesis)
# and sends ~/database.sqlite, the profiles' dataset.root.
#
# Usage (from ~/software/nn on GridUnesp, after the deploy):
#   THESIS_PHASE=phase00 sbatch --array=0-63 scripts/pipeline/thesis/thesis_gridunesp.sbatch
#   THESIS_PHASE=phase01 sbatch --array=0-95 scripts/pipeline/thesis/thesis_gridunesp.sbatch
# (one 1-CPU process at a time per task: see --cpus-per-task below for why not threads)
#   scancel --batch --signal=USR1 <jobid>_<task>   # stop a task early; its results come home
#
# ── The unit of work: a profile in Phase 00, a (profile, repetition) in Phase 01 ───────────────
# The program is fastest on ONE thread, so more CPUs help only as more processes, and the work
# has to be cut into independent pieces. The repetitions of a profile are such pieces: each is a
# whole training with its own seed. `thesis --config P --repeat k` runs repetition k exactly as
# the full run would (seed, e05_<tag>_rep<k>_* names; ThesisCli.hpp), so:
#
#   phase01  32 profiles x 3 repetitions = 96 units  -> --array=0-95, one unit per task;
#            the phase takes as long as one repetition instead of three
#   phase00  214 units, one per profile               -> --array=0-63, ~3-4 units per task
#
# Phase 00 keeps whole profiles: a handcrafted profile extracts its features ONCE for all
# repetitions (they do not depend on the seed), and split it would extract them three times; its
# slowest profile took 230 s at one thread (array 5323006). Finer than a repetition -- one process
# per outer fold -- would need a step that merges fold results into a summary, which the program
# does in memory today; not worth it for units that already take minutes.
#
# Units are interleaved over the tasks: task t of N runs units t, t+N, t+2N, ... of the sorted
# list, in a row. Every unit first loads the decoded dataset (~/database.sqlite.e05dscache, 1.6 GB)
# from /home; units that run in a row on one node read it from that node's page cache.
#
#   sorted phase00 units  p00_ae_ann_base_eeg  p00_ae_ann_base_voice  ...  p00_pca_k8_voice
#   position              0                    1                      ...  213
#   task (N = 64)         0                    1                      ...  213 % 64 = 21
#
# The array must be 0..N-1 with nothing left out (checked): the mapping is a function of N, so a
# partial resubmission (--array=3,7) would hand task 3 another task's units. Resubmit the whole
# array instead -- a task whose units are all done exits in seconds.
#
# ── Done markers, and why a marker carries the code revision ─────────────────────────────────
# A unit is done when the binary exited 0 AND every file its repetition(s) must write exists;
# then results/thesis/<phase>/.done/<unit>.done is written (<unit> = <profile> in Phase 00,
# <profile>.rep<k> in Phase 01), holding the deploy's revision label (commit/tree hash, see
# scripts/pipeline/meeting01/_provenance.sh) and the thread count. A resubmitted task skips its
# units that are done. A marker of ANOTHER label anywhere in that directory refuses the task: the
# phase would otherwise mix results of two binaries -- what happened to the July run, whose
# Phase 01 numbers came from code later found defective -- and nothing downstream could tell
# them apart. Move the old directory aside to run a new revision.
#
# ── Failures are not skipped ─────────────────────────────────────────────────────────────────
# A unit that fails (non-zero exit, or an expected file missing afterwards) stops its task at once:
# the task exits non-zero, results/thesis/<phase>/.failed/<unit>.failed holds the exit code and
# the log path, and the remaining units of that task do not run. Nothing is retried, nothing
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
# ONE CPU, and the parallelism comes from the array instead. Measured on node019, 2026-11-12
# (job 5322998: one Phase 01 profile at 1/2/4 threads, 1 and 3 epochs, the slope being the cost of
# an epoch without loading and extraction):
#
#   p01_dsnn_voice_dep_flat_raw    per epoch  1 thread 2 s    2 threads 2.5 s   4 threads 53 s
#   p01_dsnn_eeg_dep_nested_raw    per epoch  1 thread 10.5 s                   4 threads 14.5 s
#
# The threads split each small multiply (xtensor's OpenMP, OpenBLAS) and pay more in coordination
# than they save -- meeting01's finding (.wiki/Guides/GridUnesp-Deployment.md, "Threads"). Phase 00
# at 4 threads showed it too: the 64-wide "base" autoencoders took minutes per repeat where the
# 32-wide ones took seconds. And the thread count CHANGES THE RESULT: the same profile gave EER
# 0.4176 / 0.4083 / 0.4201 at 1 / 2 / 4 threads after 3 epochs (summation order). So it is part of
# the run's identity: the .done marker records it, and one phase never mixes two thread counts.
#SBATCH --cpus-per-task=1
# The decoded dataset alone is 1.6 GB in memory; the voice SNN-AE profiles peaked at ~4.4 GB RSS
# locally. Out of memory is a loud failure (the task is killed and has no marker).
#SBATCH --mem=8G
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
  *) refuse "THESIS_PHASE must be phase00 or phase01 (got '${PHASE}'): THESIS_PHASE=phase00 sbatch --array=0-63 $0" ;;
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
# The label a .done marker carries: the code AND the thread count, which changes the numbers
# (see the --cpus-per-task comment above).
REVISION="$(provenance_label SOURCE_REVISION) threads=${OMP_NUM_THREADS}"

PROFILE_DIR="src/experiments/thesis/profiles/${PHASE}"
mapfile -t ALL_PROFILES < <(find "$PROFILE_DIR" -maxdepth 1 -name '*.json' | sort)
(( ${#ALL_PROFILES[@]} > 0 )) || refuse "no profiles in $PWD/$PROFILE_DIR"

# The units (see the header): one line each, "<profile path><TAB><repetition, or - for all>".
case "$PHASE" in
  phase00) SPLIT_REPEATS=0 ;;
  phase01) SPLIT_REPEATS=1 ;;
esac
UNITS_FILE="$(mktemp)"
# Through a file, not `mapfile < <(python3 ...)`: a process substitution's exit status is lost,
# and a profile the listing refuses would silently drop its units.
python3 - "$SPLIT_REPEATS" "${ALL_PROFILES[@]}" > "$UNITS_FILE" <<'PY' ||
import json, sys
split = sys.argv[1] == "1"
for path in sys.argv[2:]:
    reps = json.load(open(path))["experiment"]["repeats"]
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        sys.exit(f"{path}: experiment.repeats must be a positive integer, got {reps!r}")
    for k in (range(reps) if split else ["-"]):
        print(f"{path}\t{k}")
PY
  { rm -f "$UNITS_FILE"; refuse "could not list the units of $PWD/$PROFILE_DIR (above)"; }
mapfile -t ALL_UNITS < "$UNITS_FILE"
rm -f "$UNITS_FILE"
(( NTASKS <= ${#ALL_UNITS[@]} )) ||
  refuse "$NTASKS tasks for ${#ALL_UNITS[@]} units: some tasks would have nothing to run"
# A unit's name in markers and logs: <profile stem>, or <profile stem>.rep<k>.
unit_id() {
  local stem
  stem="$(basename "$1" .json)"
  if [[ "$2" == "-" ]]; then echo "$stem"; else echo "${stem}.rep$2"; fi
}

RES="results/thesis/${PHASE}"
DONE_DIR="${RES}/.done"
FAILED_DIR="${RES}/.failed"

# A marker of another revision anywhere in the phase: refuse (see the header).
if [[ -d "$DONE_DIR" ]]; then
  while IFS= read -r -d '' marker; do
    if [[ "$(<"$marker")" != "$REVISION" ]]; then
      refuse "$PWD/$marker was made by '$(<"$marker")'; this task is '$REVISION'. One" \
             "phase directory must not mix two binaries' or two thread counts' results: move $PWD/$RES aside" \
             "(mv $RES ${RES}_$(date +%Y%m%d)) and resubmit the whole array."
    fi
  done < <(find "$DONE_DIR" -type f -name '*.done' -print0)   # -type f: ".done" itself matches
fi

MINE=()
for (( i = TASK; i < ${#ALL_UNITS[@]}; i += NTASKS )); do
  IFS=$'\t' read -r unit_profile unit_rep <<< "${ALL_UNITS[$i]}"
  [[ -f "${DONE_DIR}/$(unit_id "$unit_profile" "$unit_rep").done" ]] && continue
  MINE+=("${ALL_UNITS[$i]}")
done
echo "[thesis-grid] ${PHASE} task ${TASK}/${NTASKS}, revision ${REVISION}:" \
     "${#MINE[@]} unit(s) to run"
(( ${#MINE[@]} > 0 )) || { echo "[thesis-grid] every unit of this task is done"; exit 0; }
# A failure marker of one of MY units is from an earlier attempt of this task: this attempt
# reruns that unit, so it no longer describes the state.
for u in "${MINE[@]}"; do
  IFS=$'\t' read -r unit_profile unit_rep <<< "$u"
  rm -f "${FAILED_DIR}/$(unit_id "$unit_profile" "$unit_rep").failed"
done

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
for u in "${MINE[@]}"; do cp -a "${u%%$'\t'*}" "$STAGE/${PROFILE_DIR}/"; done
printf '%s\n' "${MINE[@]}" > "$STAGE/units.txt"
printf '%s\n' "$REVISION" > "$STAGE/revision.txt"

# run.sh: runs this task's units in order, from WORK_DIR, stopping at the first failure.
cat > "$STAGE/run.sh" <<'RUN'
#!/bin/bash
# run.sh -- written by thesis_gridunesp.sbatch for job-nanny; not meant to be run by hand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BIN="out/build/max-performance/src/experiments/thesis/thesis"
REVISION="$(<revision.txt)"
while IFS=$'\t' read -r profile rep; do
  stem="$(basename "$profile" .json)"
  if [[ "$rep" == "-" ]]; then unit="$stem"; args=(); else unit="${stem}.rep${rep}"; args=(--repeat "$rep"); fi
  res="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["results_dir"])' "$profile")"
  log="${res}/logs/${unit}.log"
  mkdir -p "${res}/logs" "${res}/.done" "${res}/.failed"
  echo "[thesis-grid] $(date '+%F %T') start ${unit}"
  start=$SECONDS
  rc=0
  "$BIN" --config "$profile" "${args[@]}" > "$log" 2>&1 || rc=$?
  # Every file the unit's repetitions always write (thesis.cpp: summary and paraconsistent CSV,
  # whatever the phase): e05_<run_tag>_<kind> when the profile has one repetition,
  # e05_<run_tag>_rep<k>_<kind> otherwise -- the writers add "e05_" to a run_tag that already
  # starts with it, the rule 01_thesis_phase00_rank.py's expected_csvs() builds its names from.
  missing="$(python3 - "$profile" "$res" "$rep" <<'PY'
import json, os, sys
p = json.load(open(sys.argv[1]))
tag, reps = p["experiment"]["run_tag"], p["experiment"]["repeats"]
ks = range(reps) if sys.argv[3] == "-" else [int(sys.argv[3])]
tags = [tag] if reps == 1 else [f"{tag}_rep{k}" for k in ks]
names = [f"e05_{t}_{kind}" for t in tags for kind in ("summary.json", "paraconsistent.csv")]
print(" ".join(n for n in names if not os.path.isfile(os.path.join(sys.argv[2], n))))
PY
)"
  if [[ "$rc" != 0 || -n "$missing" ]]; then
    printf 'exit=%s\nmissing=%s\nlog=%s\nrevision=%s\n' "$rc" "$missing" "$log" "$REVISION" \
      > "${res}/.failed/${unit}.failed"
    echo "[thesis-grid] FAILED ${unit}: exit ${rc}${missing:+, missing ${missing}}; last lines of ${log}:" >&2
    tail -c 4000 "$log" | tr '\r' '\n' | tail -n 30 >&2
    exit 1
  fi
  printf '%s\n' "$REVISION" > "${res}/.done/${unit}.done"
  echo "[thesis-grid] $(date '+%F %T') done ${unit} ($(( SECONDS - start )) s)"
done < units.txt
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
