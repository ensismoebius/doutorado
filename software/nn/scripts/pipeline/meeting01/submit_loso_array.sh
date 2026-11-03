#!/usr/bin/env bash
# submit_loso_array.sh -- submit (or resubmit) the nested-LOSO run as a Slurm job ARRAY: one
# one-CPU task per (dataset, outer fold), 24 in all. Run on GridUnesp from ~/software/nn, after
# gridunesp_deploy.sh. Why an array and why 1 CPU: the measured cost of this program is per
# thread (the wiki's "Threads" section: 20 threads ran ~1000x slower than 1), so the folds run
# side by side instead of in series.
#
# Task t = dataset (fsdd audiomnist eegmmidb siena)[t / 6], fold t % 6 -- the mapping the sbatch
# derives and checks against the profile.
#
# It submits ONLY the unfinished tasks: a task is finished when its fold's
# results/meeting01/meeting01_loso_<dataset>_fold<f>_comparative_metrics.csv exists and is
# non-empty (the file the run writes last, through a rename). So after a time limit, a scancel or a
# node failure, running this again is the whole recovery: finished folds are not repeated, a fold
# that was cut short is rerun (RESUME=1): its finished runs are recomputed, and only the search
# that was cut mid-way continues from its stamped GA checkpoint (see 01_meeting01_run_loso.sh).
#
# It REFUSES while any meeting01-loso* job of yours is RUNNING or PENDING, so that a resubmission
# cannot run a fold twice; cancel first (scancel --batch --signal=USR1 <id> copies results back).
#
# Usage (from ~/software/nn on GridUnesp):
#   scripts/pipeline/meeting01/submit_loso_array.sh --dry-run          # show what it would submit
#   scripts/pipeline/meeting01/submit_loso_array.sh                    # submit the unfinished tasks
#   scripts/pipeline/meeting01/submit_loso_array.sh --only "0 7"       # just these tasks
#   --partition P (default long)   --time D-HH:MM:SS (default 30-00:00:00)
#   --max-parallel N               at most N tasks at once (the array's %N throttle)
#   --extra "ARGS"                 passed to sbatch as is, e.g. a smoke test:
#       --partition short --time 00:06:00 --extra "--signal=B:USR1@150 --job-name=meeting01-loso-smoke"
#
# Not run by hand elsewhere: the real run is weeks-scale per fold; this script only submits.
set -euo pipefail

SBATCH_FILE="scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch"
RES="results/meeting01"
DATASETS=(fsdd audiomnist eegmmidb siena)
FOLDS=6

dry=0 partition="long" time="30-00:00:00" only="" max_par="" extra=""
while (($#)); do
  case "$1" in
    --dry-run) dry=1 ;;
    --partition) partition="$2"; shift ;;
    --time) time="$2"; shift ;;
    --only) only="$2"; shift ;;
    --max-parallel) max_par="$2"; shift ;;
    --extra) extra="$2"; shift ;;
    *) echo "submit_loso_array.sh: unknown argument '$1' (see the header)" >&2; exit 2 ;;
  esac
  shift
done

[[ -f "$SBATCH_FILE" ]] || { echo "REFUSED: run from ~/software/nn ($SBATCH_FILE not found)" >&2; exit 1; }
[[ -z "$max_par" || "$max_par" =~ ^[0-9]+$ ]] || { echo "--max-parallel needs an integer" >&2; exit 2; }

live="$(squeue -u "$USER" -h -t RUNNING,COMPLETING,PENDING -o '%i %j %T' | awk '$2 ~ /^meeting01-loso/')"
if [[ -n "$live" ]]; then
  echo "REFUSED: meeting01-loso job(s) already queued or running -- resubmitting could run a fold" \
       "twice:" >&2
  echo "$live" | sed 's/^/    /' >&2
  echo "Wait for them, or stop them (scancel --batch --signal=USR1 <id>), then run this again." >&2
  exit 1
fi

tasks=()
for ((t = 0; t < ${#DATASETS[@]} * FOLDS; t++)); do
  ds="${DATASETS[$((t / FOLDS))]}" f=$((t % FOLDS))
  if [[ -n "$only" ]]; then
    [[ " $only " == *" $t "* ]] || continue
  fi
  if [[ -s "$RES/meeting01_loso_${ds}_fold${f}_comparative_metrics.csv" ]]; then
    echo "task $t ($ds fold $f): finished, not resubmitted"
    continue
  fi
  tasks+=("$t")
done
if ((${#tasks[@]} == 0)); then
  echo "nothing to submit: every selected task has its comparative_metrics.csv"
  exit 0
fi

# A first submission has an empty results dir and runs fresh (the sbatch's own guard then checks
# each task's fold). Once anything exists, RESUME=1: finished folds are skipped, a cut-short fold
# continues from its checkpoints, an untouched one simply runs.
resume=0
if [[ -d "$RES" && -n "$(ls -A "$RES")" ]]; then resume=1; fi

list="$(IFS=,; echo "${tasks[*]}")"
[[ -z "$max_par" ]] || list="${list}%${max_par}"
cmd=(sbatch --array="$list" -c 1 -p "$partition" -t "$time")
# shellcheck disable=SC2206
[[ -z "$extra" ]] || cmd+=($extra)
cmd+=("$SBATCH_FILE")

echo "tasks to submit: ${tasks[*]}   (RESUME=$resume)"
echo "+ RESUME=$resume ${cmd[*]}"
((dry)) && exit 0
if ((resume)); then RESUME=1 "${cmd[@]}"; else env -u RESUME "${cmd[@]}"; fi
