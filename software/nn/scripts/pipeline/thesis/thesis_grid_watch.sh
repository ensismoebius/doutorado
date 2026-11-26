#!/usr/bin/env bash
# thesis_grid_watch.sh -- watch the thesis Slurm arrays on GridUnesp, refreshed in place.
#
# Runs scripts/pipeline/thesis/thesis_grid_status.py ON the login node through ONE ssh session:
# the script is sent over ssh's stdin (`python3 -`), so the cluster does not need a redeploy to
# get the current version -- a redeploy would change the deploy revision, which is part of the
# per-fold feature cache's key and of the phase markers. The refresh loop runs inside that one
# session: GridUnesp's Fail2Ban locks repeated logins for 15 minutes
# (.wiki/Guides/GridUnesp-Deployment.md), so never wrap this script in a loop of your own.
#
# Usage (from software/nn; credentials from scripts/pipeline/meeting01/.env, see
# _gridunesp_env.sh):
#   scripts/pipeline/thesis/thesis_grid_watch.sh                     # refresh every 60 s
#   scripts/pipeline/thesis/thesis_grid_watch.sh --interval 120
#   scripts/pipeline/thesis/thesis_grid_watch.sh --once              # one frame, then exit
#   scripts/pipeline/thesis/thesis_grid_watch.sh --jobs 5345042 --since-hours 96
# Every argument goes to thesis_grid_status.py (its --help lists them). Ctrl-C to quit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../meeting01/_gridunesp_env.sh
source "${HERE}/../meeting01/_gridunesp_env.sh"

HOST="${GRIDUNESP_HOST:-access.grid.unesp.br}"
REMOTE_DIR="${GRIDUNESP_REMOTE_DIR:-software/nn}"

args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done

# The cluster's own python3 (3.6.8, stdlib only) runs it: no module, no conda activation.
exec sshpass -e ssh "${GRIDUNESP_USER}@${HOST}" \
  "cd ${REMOTE_DIR} && python3 -u -${args}" < "${HERE}/thesis_grid_status.py"
