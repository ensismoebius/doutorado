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
#
# What gets synced: everything under this checkout EXCEPT out/ (local build
# artifacts -- the remote builds its own, compiled for its own CPU) and results/
# (in-progress or completed run output), and scripts/pipeline/meeting01/.env (your
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

echo "[gridunesp-deploy] ensuring remote dir ${REMOTE_DIR} exists"
ssh -o ControlPath="$CTRL_PATH" "${GRIDUNESP_USER}@${HOST}" "mkdir -p '${REMOTE_DIR}'"

echo "[gridunesp-deploy] syncing checkout to ${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}"
rsync -avz --delete --info=progress2 -e "ssh -o ControlPath=${CTRL_PATH}" \
  --exclude out/ --exclude results/ --exclude '*.o' --exclude '__pycache__/' \
  --exclude '.venv/' --exclude /scripts/pipeline/meeting01/.env \
  "$ROOT_DIR/" "${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}/"

# The revision goes AFTER the tree on purpose: the --delete above removed any older copy (it
# is not in this checkout), so a deploy that dies between the two steps leaves the cluster
# with NO SOURCE_REVISION -- which 01_meeting01_run_loso.sh refuses -- never with a stale one.
echo "[gridunesp-deploy] shipping SOURCE_REVISION (${REVISION})"
rsync -a -e "ssh -o ControlPath=${CTRL_PATH}" "$REV_FILE" \
  "${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}/SOURCE_REVISION"

echo "[gridunesp-deploy] remote environment + datasets + configure + build (reusing the same connection)"
ssh -o ControlPath="$CTRL_PATH" "${GRIDUNESP_USER}@${HOST}" bash -s -- "$REMOTE_DIR" "$BUILD_CPUS" "${SKIP_DATASETS:-0}" <<'REMOTE'
set -euo pipefail
cd "$1"
BUILD_CPUS="$2"
SKIP_DATASETS="$3"

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

if [[ "$SKIP_DATASETS" == "1" ]]; then
  echo "[gridunesp-deploy:remote] datasets: SKIPPED (SKIP_DATASETS=1) -- only when an earlier" \
       "deploy already completed ensure_datasets.sh; the profile's roots are not re-verified"
else
  echo "[gridunesp-deploy:remote] datasets (skips anything already present)"
  ./scripts/pipeline/meeting01/ensure_datasets.sh
fi

echo "[gridunesp-deploy:remote] configuring (login node, needs internet for FetchContent)"
cmake --preset=max-performance

# A stamp from an earlier deploy must not survive a deploy that then fails to build.
rm -f out/build/max-performance/SOURCE_REVISION.built

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
    tail -n 40 "$log" >&2 || true
    return 1
  fi
}

echo "[gridunesp-deploy:remote] building on a compute node (sbatch --wait, cpus=${BUILD_CPUS})"
on_compute_node build "$BUILD_CPUS" 00:30:00 '
  module load miniconda/24.4.0-libmamba
  eval "$(conda shell.bash hook)"
  conda activate meeting01-build
  cmake --build out/build/max-performance --target meeting01 -j'"$BUILD_CPUS"'
'

# The job's exit code is the build's (set -e in the job script, sbatch --wait returns it), so
# a failed build never gets here. NOT a `ninja -n` "no work to do" check: with CMake's globbed
# directories a dry run always reports "Re-running CMake" (it cannot restat), measured on the
# cluster 2026-10-07 right after a build that had worked.
BIN=out/build/max-performance/src/experiments/meeting01/meeting01
[[ -x "$BIN" ]] || { echo "[gridunesp-deploy:remote] ERROR: no binary at $BIN" >&2; exit 1; }

echo "[gridunesp-deploy:remote] smoke check"
on_compute_node smoke 4 00:10:00 "$BIN --help >/dev/null"

# Only a binary that built AND ran gets the stamp, so "stamp == SOURCE_REVISION" means
# "this binary was built from exactly the sources on disk".
cp SOURCE_REVISION out/build/max-performance/SOURCE_REVISION.built
echo "[gridunesp-deploy:remote] build OK, stamped with $(sed -n 's/^commit=//p' SOURCE_REVISION)"

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
