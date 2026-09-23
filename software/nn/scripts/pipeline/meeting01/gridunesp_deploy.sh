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
#   GRIDUNESP_BUILD_CPUS  default 26, not the node's full 28 -- ~2 cores/node are
#                         OS-reserved per the v3 manual (2026-09-23 check), so
#                         requesting 28 risks the build's srun allocation hanging in
#                         PENDING instead of starting.
#
# What gets synced: everything under this checkout EXCEPT out/ (local build
# artifacts -- the remote builds its own, compiled for its own CPU) and results/
# (in-progress or completed run output). rsync's --delete keeps the remote source
# tree an exact mirror of this one, but --delete never touches an excluded path, so
# a redeploy can never wipe a run already in progress on the remote.
set -euo pipefail

# shellcheck source=./_gridunesp_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gridunesp_env.sh"

HOST="${GRIDUNESP_HOST:-access.grid.unesp.br}"
REMOTE_DIR="${GRIDUNESP_REMOTE_DIR:-software/nn}"
BUILD_CPUS="${GRIDUNESP_BUILD_CPUS:-26}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

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

echo "[gridunesp-deploy] syncing checkout to ${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}"
rsync -avz --delete --info=progress2 -e "ssh -o ControlPath=${CTRL_PATH}" \
  --exclude out/ --exclude results/ --exclude '*.o' --exclude '__pycache__/' \
  --exclude '.venv/' \
  "$ROOT_DIR/" "${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}/"

echo "[gridunesp-deploy] remote environment + datasets + configure + build (reusing the same connection)"
ssh -o ControlPath="$CTRL_PATH" "${GRIDUNESP_USER}@${HOST}" bash -s -- "$REMOTE_DIR" "$BUILD_CPUS" <<'REMOTE'
set -euo pipefail
cd "$1"
BUILD_CPUS="$2"

echo "[gridunesp-deploy:remote] toolchain env (module load + conda create/update)"
module load miniconda/24.4.0-libmamba
./scripts/pipeline/meeting01/gridunesp_setup_env.sh
conda activate meeting01-build

echo "[gridunesp-deploy:remote] datasets (skips anything already present)"
./scripts/pipeline/meeting01/ensure_datasets.sh

echo "[gridunesp-deploy:remote] configuring (login node, needs internet for FetchContent)"
cmake --preset=max-performance

echo "[gridunesp-deploy:remote] building on a compute node (srun, cpus=${BUILD_CPUS})"
srun --partition=short --time=00:30:00 --cpus-per-task="$BUILD_CPUS" bash -c "
  module load miniconda/24.4.0-libmamba
  conda activate meeting01-build
  cmake --build out/build/max-performance --target meeting01 -j${BUILD_CPUS}
"

echo "[gridunesp-deploy:remote] smoke check"
srun --partition=short --time=00:10:00 --cpus-per-task=4 \
  out/build/max-performance/src/experiments/meeting01/meeting01 --help >/dev/null
echo "[gridunesp-deploy:remote] build OK"
REMOTE

cat <<EOF

[gridunesp-deploy] done. Environment, all 4 datasets, and the built binary are ready
on ${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}. Nothing has been submitted to the queue
yet -- that is this script's one deliberate stop.

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
