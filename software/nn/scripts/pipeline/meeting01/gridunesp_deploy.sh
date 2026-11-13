#!/usr/bin/env bash
# gridunesp_deploy.sh — one-shot installer: pushes this checkout to GridUnesp,
# bootstraps the conda toolchain, fetches all 4 datasets, configures and builds the
# meeting01 binary -- everything needed before the real run. Safe to re-run any time
# (rsync is incremental, env setup and dataset fetch are both idempotent).
#
# Deliberately does NOT submit the actual training job: it prints the exact `sbatch`
# command at the end instead and stops. The real run is weeks-to-months of compute
# (see meeting01-loso.json's _total_runs_breakdown / .wiki/Guides/
# GridUnesp-Deployment.md) -- starting it is a decision for a human, not something
# this script makes on your behalf.
#
# Usage (from the local machine, in software/nn; needs a GridUnesp account already
# approved -- .wiki/Guides/GridUnesp-Deployment.md Sec 0). First run prompts for
# username + password and saves them to .env next to this script (see
# _gridunesp_env.sh); later runs read .env instead of asking again:
#   ./scripts/pipeline/meeting01/gridunesp_deploy.sh
#
# Env overrides:
#   GRIDUNESP_HOST        default access.grid.unesp.br
#   GRIDUNESP_REMOTE_DIR  default software/nn (relative to the remote $HOME)
#   SKIP_DATASETS=1       skip ensure_datasets.sh (only after a deploy that completed it)
#   GRIDUNESP_BUILD_CPUS  default 26, not the node's full 28 -- ~2 cores/node are
#                         OS-reserved per the v3 manual (2026-09-23 check), so
#                         requesting 28 risks the build's srun allocation hanging in
#                         PENDING instead of starting.
#   GRIDUNESP_TARGETS     default "meeting01 thesis" -- the binaries to build, smoke and
#                         stamp. Each binary has its own stamp (meeting01:
#                         SOURCE_REVISION.built, thesis: SOURCE_REVISION.thesis.built), and
#                         EVERY stamp is removed before the build: a binary this deploy did
#                         not rebuild is left unstamped, so its runner refuses to start
#                         instead of running old code under the new SOURCE_REVISION.
#
# thesis: its profiles read ~/database.sqlite (dataset.root), so with `thesis` in the targets
# the local ~/database.sqlite (2.3 GB) is sent to the same path on the cluster, with its
# decoded-dataset caches when there are some: one per EEG condition,
# ~/database.sqlite.pronounced.e05dscache and ~/database.sqlite.imagined-<seed>.e05dscache
# (ThesisDataset.cpp; rsync -a keeps the mtime the caches' signature is computed from). The
# pre-2026-11-12 ~/database.sqlite.e05dscache, decoded with the previous-trial EEG, is never
# read again and is not sent. The smoke for `thesis` is two real Phase 00 profiles (PCA, k = 8,
# EEG of pronounced and of imagined speech) on a compute node: they prove the binary, the
# database and both caches, and leave valid caches behind before an array starts. The jobs are
# submitted with scripts/pipeline/thesis/thesis_gridunesp.sbatch (named thesis-run-*).
#
# What gets synced: everything under this checkout EXCEPT out/ (local build
# artifacts -- the remote builds its own, compiled for its own CPU) and results/
# (in-progress or completed run output), the job logs and job-nanny staging directories the
# sbatch scripts leave in the tree root (meeting01_loso_<id>.out/.err, thesis_run_<id>_<t>.out/.err,
# .job-input-<id>/), which
# --delete would otherwise wipe on every deploy, and scripts/pipeline/meeting01/.env (your
# GridUnesp password, which nothing on the cluster reads; it used to be copied too). rsync's
# --delete keeps the remote source tree an exact mirror of this one, but --delete never
# touches an excluded path, so a redeploy can never wipe a run already in progress on the
# remote.
#
# Which code is on the cluster: before anything is sent, source_revision.py writes the
# tree's revision (commit, dirty flag, hash of every source file's content) and the deploy
# ships it as SOURCE_REVISION AFTER the tree; after a successful build it is copied to
# out/build/max-performance/SOURCE_REVISION.built. 01_meeting01_run_loso.sh refuses to start
# when the two differ -- a sync whose build failed would otherwise run the old binary under
# the new revision. See scripts/pipeline/meeting01/_provenance.sh.
set -euo pipefail

# shellcheck source=./_gridunesp_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gridunesp_env.sh"

HOST="${GRIDUNESP_HOST:-access.grid.unesp.br}"
REMOTE_DIR="${GRIDUNESP_REMOTE_DIR:-software/nn}"
BUILD_CPUS="${GRIDUNESP_BUILD_CPUS:-26}"
TARGETS="${GRIDUNESP_TARGETS:-meeting01 thesis}"
read -r -a TARGET_LIST <<< "$TARGETS"
[[ ${#TARGET_LIST[@]} -gt 0 ]] || { echo "gridunesp_deploy.sh: GRIDUNESP_TARGETS is empty" >&2; exit 1; }
for t in "${TARGET_LIST[@]}"; do
  case "$t" in
    meeting01|thesis) ;;
    *) echo "gridunesp_deploy.sh: unknown target '$t' in GRIDUNESP_TARGETS (known: meeting01 thesis)" >&2
       exit 1 ;;
  esac
done
has_target() { [[ " ${TARGET_LIST[*]} " == *" $1 "* ]]; }
# ssh joins its arguments into ONE command line that the remote shell splits again, so a
# space-separated list arrives as several arguments (the first thesis deploy, 2026-11-12, built
# meeting01 alone and still printed "thesis (built and smoked)"). The list travels as one
# comma-joined word instead, and the remote side checks it got exactly the four it expects.
TARGET_CSV="$(IFS=,; echo "${TARGET_LIST[*]}")"
THESIS_DB="${HOME}/database.sqlite"
if has_target thesis && [[ ! -f "$THESIS_DB" ]]; then
  echo "gridunesp_deploy.sh: target 'thesis' needs ${THESIS_DB} (the thesis profiles'" \
       "dataset.root), and it does not exist here" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=./_provenance.sh
source "$ROOT_DIR/scripts/pipeline/meeting01/_provenance.sh"

# Which code is about to be shipped? Worked out before the first ssh, so a tree that cannot
# be identified (not a git checkout, no commit yet) fails without touching the login node.
REV_FILE="$(mktemp)"
python3 "$ROOT_DIR/scripts/pipeline/meeting01/source_revision.py" compute "$ROOT_DIR" \
  > "$REV_FILE"
chmod 644 "$REV_FILE"
REVISION="$(provenance_label "$REV_FILE")"
echo "[gridunesp-deploy] source revision: ${REVISION}"

# One multiplexed SSH connection for the whole script: the first call below is what
# actually authenticates (via sshpass -e, using the password _gridunesp_env.sh just
# loaded/prompted) and every later ssh/rsync call below reuses that same connection
# over CTRL_PATH instead of authenticating again. Also friendlier to the login
# node's Fail2Ban lockout -- one real connection attempt instead of three.
CTRL_DIR="$(mktemp -d)"
CTRL_PATH="${CTRL_DIR}/ssh-%r@%h:%p"
cleanup() {
  ssh -o ControlPath="$CTRL_PATH" -O exit "${GRIDUNESP_USER}@${HOST}" >/dev/null 2>&1 || true
  rm -rf "$CTRL_DIR"
  rm -f "$REV_FILE"
}
trap cleanup EXIT

echo "[gridunesp-deploy] connecting to ${GRIDUNESP_USER}@${HOST}"
sshpass -e ssh -o ControlMaster=auto -o ControlPath="$CTRL_PATH" -o ControlPersist=15m \
    -o ConnectTimeout=15 "${GRIDUNESP_USER}@${HOST}" true || {
  echo "gridunesp_deploy.sh: could not reach ${GRIDUNESP_USER}@${HOST} -- confirm your" \
       "account is approved and that the password in" \
       "$(dirname "${BASH_SOURCE[0]}")/.env is still correct (delete that file to be" \
       "re-prompted), then try again. Avoid retrying rapidly: repeated failed" \
       "connections trigger a 15-minute Fail2Ban lockout (see" \
       ".wiki/Guides/GridUnesp-Deployment.md)." >&2
  exit 1
}

# Not while a run is going on. The job stages only the meeting01 EXECUTABLE; the shared
# libraries it needs from this project (libstatistics.so, libmatio*) are loaded from the remote
# build tree -- an absolute RPATH -- by every fold's process, each fold being a new process. A
# rebuild under a running job would hand its later folds different code, recorded nowhere
# (source_revisions.log is written once per start). A PENDING job is fine: it stages whatever is
# deployed when it starts, and logs that revision. Jobs are recognised by the meeting01-loso
# prefix of their name (the sbatch's --job-name; keep it in any override). thesis jobs
# (thesis-run-*) load the same libraries the same way, one process per profile, and also read
# ~/database.sqlite, which this deploy may replace: they block a deploy too.
if ! RUNNING_JOBS="$(ssh -o ControlPath="$CTRL_PATH" "${GRIDUNESP_USER}@${HOST}" \
      "set -o pipefail; squeue -u \"\$USER\" -h -t RUNNING,COMPLETING -o '%i %j %M' | awk '\$2 ~ /^(meeting01-loso|thesis-run)/'")"
then
  echo "gridunesp_deploy.sh: could not list your jobs on ${HOST} (squeue failed); not deploying" \
       "without knowing whether a run would be disturbed." >&2
  exit 1
fi
if [[ -n "$RUNNING_JOBS" ]]; then
  echo "gridunesp_deploy.sh: REFUSED -- a meeting01-loso or thesis-run job is running from ${REMOTE_DIR}:" >&2
  echo "$RUNNING_JOBS" | sed 's/^/    /' >&2
  echo "Its folds load shared libraries from the build tree this deploy would rebuild. Wait for" \
       "it to end, or stop it first -- 'scancel --batch --signal=USR1 <jobid>' copies its" \
       "results back (see .wiki/Guides/GridUnesp-Deployment.md)." >&2
  exit 1
fi

echo "[gridunesp-deploy] ensuring remote dir ${REMOTE_DIR} exists"
ssh -o ControlPath="$CTRL_PATH" "${GRIDUNESP_USER}@${HOST}" "mkdir -p '${REMOTE_DIR}'"

echo "[gridunesp-deploy] syncing checkout to ${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}"
rsync -avz --delete --info=progress2 -e "ssh -o ControlPath=${CTRL_PATH}" \
  --exclude out/ --exclude results/ --exclude '*.o' --exclude '__pycache__/' \
  --exclude '.venv/' --exclude /scripts/pipeline/meeting01/.env \
  --exclude '/meeting01_loso_*.out' --exclude '/meeting01_loso_*.err' --exclude '/.job-input-*/' \
  --exclude '/thesis_run_*.out' --exclude '/thesis_run_*.err' \
  "$ROOT_DIR/" "${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}/"

# The revision goes AFTER the tree on purpose: the --delete above removed any older copy (it
# is not in this checkout), so a deploy that dies between the two steps leaves the cluster
# with NO SOURCE_REVISION -- which 01_meeting01_run_loso.sh refuses -- never with a stale one.
echo "[gridunesp-deploy] shipping SOURCE_REVISION (${REVISION})"
rsync -a -e "ssh -o ControlPath=${CTRL_PATH}" "$REV_FILE" \
  "${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}/SOURCE_REVISION"

if has_target thesis; then
  thesis_files=("$THESIS_DB")
  for cache in "${THESIS_DB}".pronounced.e05dscache "${THESIS_DB}".imagined-*.e05dscache; do
    [[ -f "$cache" ]] && thesis_files+=("$cache")
  done
  # BIOMEX-DB (the profiles under profiles/biomex, dataset.source biomex2021) is a separate
  # base, written by scripts/data/biomex_to_sqlite.py. Sent when it exists; when it does not,
  # say so: a biomex array on the cluster would then refuse to open it, not run without it.
  if [[ -f "${HOME}/biomex.sqlite" ]]; then
    thesis_files+=("${HOME}/biomex.sqlite")
  else
    echo "[gridunesp-deploy] ${HOME}/biomex.sqlite does not exist here: NOT sent (THESIS_PHASE=biomex will refuse on the cluster)"
  fi
  echo "[gridunesp-deploy] thesis dataset -> ~/ on the cluster: ${thesis_files[*]}"
  rsync -a --info=progress2 -e "ssh -o ControlPath=${CTRL_PATH}" "${thesis_files[@]}" \
    "${GRIDUNESP_USER}@${HOST}:"
fi

echo "[gridunesp-deploy] remote environment + datasets + configure + build (reusing the same connection)"
ssh -o ControlPath="$CTRL_PATH" "${GRIDUNESP_USER}@${HOST}" bash -s -- "$REMOTE_DIR" "$BUILD_CPUS" "${SKIP_DATASETS:-0}" "${TARGET_CSV}" <<'REMOTE'
set -euo pipefail
cd "$1"
BUILD_CPUS="$2"
SKIP_DATASETS="$3"
IFS=, read -r -a TARGET_LIST <<< "$4"
[[ $# -eq 4 && ${#TARGET_LIST[@]} -gt 0 ]] ||
  { echo "[gridunesp-deploy:remote] ERROR: expected 4 arguments, the last a comma-separated target list; got $# ($*)" >&2; exit 1; }
has_target() { [[ " ${TARGET_LIST[*]} " == *" $1 "* ]]; }

echo "[gridunesp-deploy:remote] toolchain env (module load + conda create/update)"
module load miniconda/24.4.0-libmamba
./scripts/pipeline/meeting01/gridunesp_setup_env.sh
# `module load` only puts the conda binary on PATH -- it does not run `conda init`,
# so the `conda activate` shell function does not exist yet in this non-interactive
# `ssh ... bash -s` session (only an interactive login shell that has run `conda
# init` gets it). Sourcing the hook here does what `conda init` would have done,
# scoped to just this script.
eval "$(conda shell.bash hook)"
conda activate meeting01-build

if ! has_target meeting01; then
  echo "[gridunesp-deploy:remote] datasets: not needed (meeting01 is not a target)"
elif [[ "$SKIP_DATASETS" == "1" ]]; then
  echo "[gridunesp-deploy:remote] datasets: SKIPPED (SKIP_DATASETS=1) -- only when an earlier" \
       "deploy already completed ensure_datasets.sh; the profile's roots are not re-verified"
else
  echo "[gridunesp-deploy:remote] datasets (skips anything already present)"
  ./scripts/pipeline/meeting01/ensure_datasets.sh
fi

echo "[gridunesp-deploy:remote] configuring (login node, needs internet for FetchContent)"
cmake --preset=max-performance

# A stamp from an earlier deploy must not survive a deploy that then fails to build, nor one
# that does not rebuild that binary at all (a target left out of GRIDUNESP_TARGETS).
rm -f out/build/max-performance/SOURCE_REVISION.built out/build/max-performance/SOURCE_REVISION.thesis.built

# Compute-node work goes through `sbatch --wait`, not `srun`. From the login node, in this
# non-interactive session, srun got its allocation but then timed out launching the step
# ("started 0 of N tasks", 12 minutes in, on three different healthy nodes) -- AND still
# exited 0, so a build that never ran looked like one that had. sbatch needs no step launch
# from here, is the same path the real job takes, and --wait returns the job's own exit
# code. The script goes in a file with a bash shebang (the conda hook needs bash, and
# `sbatch --wrap` would run it under sh).
on_compute_node() {   # on_compute_node NAME CPUS TIME 'script'
  local name="$1" cpus="$2" limit="$3" script="$4"
  local dir="$PWD/out/build/max-performance"
  local file="${dir}/deploy-${name}.sh" log="${dir}/deploy-${name}.log"
  mkdir -p "$dir"
  rm -f "$log"
  printf '#!/bin/bash\nset -eo pipefail\ncd %q\n%s\n' "$PWD" "$script" > "$file"
  chmod +x "$file"
  if ! sbatch --wait --partition=short --time="$limit" --cpus-per-task="$cpus" \
        --job-name="meeting01-${name}" --output="$log" "$file"
  then
    echo "[gridunesp-deploy:remote] ERROR: the ${name} job failed; last lines of ${log}:" >&2
    # The compute node wrote the log over NFS; the login node may not see it for a few seconds
    # (the first failed thesis build, 2026-11-12, printed "cannot open" and hid the error).
    local _
    for _ in {1..30}; do [[ -s "$log" ]] && break; sleep 1; done
    tail -n 40 "$log" >&2 || true
    return 1
  fi
}

echo "[gridunesp-deploy:remote] building on a compute node (sbatch --wait, cpus=${BUILD_CPUS})"
on_compute_node build "$BUILD_CPUS" 00:30:00 '
  module load miniconda/24.4.0-libmamba
  eval "$(conda shell.bash hook)"
  conda activate meeting01-build
  cmake --build out/build/max-performance --target '"${TARGET_LIST[*]}"' -j'"$BUILD_CPUS"'
'

# The job's exit code is the build's (set -e in the job script, sbatch --wait returns it), so
# a failed build never gets here. NOT a `ninja -n` "no work to do" check: with CMake's globbed
# directories a dry run always reports "Re-running CMake" (it cannot restat), measured on the
# cluster 2026-10-07 right after a build that had worked.
# Only a binary that built AND ran gets its stamp, so "stamp == SOURCE_REVISION" means
# "this binary was built from exactly the sources on disk".
if has_target meeting01; then
  BIN=out/build/max-performance/src/experiments/meeting01/meeting01
  [[ -x "$BIN" ]] || { echo "[gridunesp-deploy:remote] ERROR: no binary at $BIN" >&2; exit 1; }
  echo "[gridunesp-deploy:remote] smoke check: meeting01"
  on_compute_node smoke 4 00:10:00 "$BIN --help >/dev/null"
  cp SOURCE_REVISION out/build/max-performance/SOURCE_REVISION.built
fi

if has_target thesis; then
  TBIN="$PWD/out/build/max-performance/src/experiments/thesis/thesis"
  [[ -x "$TBIN" ]] || { echo "[gridunesp-deploy:remote] ERROR: no binary at $TBIN" >&2; exit 1; }
  # A real profile, not --help: the thesis binary has no help mode, and what can break here is
  # the database or its cache as much as the binary. Its results go to a scratch directory (the
  # profile's results_dir is relative), never into results/thesis.
  SMOKE_DIR="$PWD/out/build/max-performance/thesis-smoke"
  SMOKE_PROFILES="$PWD/src/experiments/thesis/profiles/phase00/p00_pca_k8_eegpron.json"
  SMOKE_PROFILES+=" $PWD/src/experiments/thesis/profiles/phase00/p00_pca_k8_eegimag.json"
  rm -rf "$SMOKE_DIR"
  echo "[gridunesp-deploy:remote] smoke check: thesis (PCA k = 8, eegpron + eegimag)"
  on_compute_node thesis-smoke 4 00:30:00 "
    mkdir -p '$SMOKE_DIR' && cd '$SMOKE_DIR'
    for prof in $SMOKE_PROFILES; do
      OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 '$TBIN' --config \"\$prof\" || exit 1
    done
    ls results/thesis/phase00/*eegpron*_summary.json results/thesis/phase00/*eegimag*_summary.json >/dev/null
  "
  cp SOURCE_REVISION out/build/max-performance/SOURCE_REVISION.thesis.built
fi
echo "[gridunesp-deploy:remote] build OK (${TARGET_LIST[*]}), stamped with $(sed -n 's/^commit=//p' SOURCE_REVISION)"

# Earlier deploys copied your local credentials file here. Nothing on the cluster reads it.
if [[ -e scripts/pipeline/meeting01/.env ]]; then
  echo "[gridunesp-deploy:remote] WARNING: $PWD/scripts/pipeline/meeting01/.env is a copy of" \
       "your local GridUnesp password file, shipped by an earlier deploy. Nothing here reads" \
       "it; delete it:  rm $PWD/scripts/pipeline/meeting01/.env" >&2
fi
REMOTE

cat <<EOF

[gridunesp-deploy] done. Environment, all 4 datasets, and the built binary are ready
on ${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}. Nothing has been submitted to the queue
yet -- that is this script's one deliberate stop.

Source revision on the cluster: ${REVISION}
(every fold's events file will carry it; so will results/meeting01/source_revisions.log)

To start the real run (weeks-to-months, see meeting01-loso.json's
_total_runs_breakdown; plain ssh below will prompt for your password once, same as
any other ssh login -- this is a deliberate one-off action, not something this repo
tries to make frictionless):
  ssh ${GRIDUNESP_USER}@${HOST} 'cd ${REMOTE_DIR} && sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch'

To resume an interrupted run instead:
  ssh ${GRIDUNESP_USER}@${HOST} 'cd ${REMOTE_DIR} && RESUME=1 sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch'

To watch progress once it is running (reads the saved .env automatically):
  ./scripts/pipeline/meeting01/remote_monitor.sh
EOF
if has_target thesis; then
  cat <<EOF

thesis (built and smoked): one array of 1-CPU tasks per phase, e.g. Phase 00 in 64 tasks --
  ssh ${GRIDUNESP_USER}@${HOST} 'cd ${REMOTE_DIR} && THESIS_PHASE=phase00 sbatch --array=0-63 scripts/pipeline/thesis/thesis_gridunesp.sbatch'
(scripts/pipeline/thesis/thesis_gridunesp.sbatch says how tasks, resubmission and revisions work)
EOF
fi
