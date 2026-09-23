#!/usr/bin/env bash
# pull_progress.sh — one-shot rsync of results/meeting01/ from GridUnesp to this
# machine, so the LOCAL monitor.py can be used against a synced copy -- e.g. for its
# full `rich` dashboard (if this machine's .venv has rich installed, see
# scripts/requirements.txt), or to archive/diff progress offline. remote_monitor.sh
# is the simpler choice for a quick live check; use this one when you specifically
# want the rich dashboard or a local copy of the data.
#
# Usage (first run prompts for username + password and saves them to .env next to
# this script -- see _gridunesp_env.sh; later runs read .env instead of asking
# again):
#   ./scripts/pipeline/meeting01/pull_progress.sh
#   then: .venv/bin/python3 scripts/pipeline/meeting01/monitor.py --run-tag meeting01_loso
#
# Safe to re-run repeatedly -- rsync only transfers deltas, and merges into whatever
# folds already exist locally, same semantics as the final "bringing results home"
# sync in .wiki/Guides/GridUnesp-Deployment.md Sec 5. Each run is still ONE
# connection, though: do not wrap this in a tight polling loop. GridUnesp's login
# node Fail2Ban-locks repeated rapid connections for 15 minutes; `watch -n 60` or
# similar with a real delay is fine, anything faster than ~every 30s risks tripping
# it.
set -euo pipefail

# shellcheck source=./_gridunesp_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gridunesp_env.sh"

HOST="${GRIDUNESP_HOST:-access.grid.unesp.br}"
REMOTE_DIR="${GRIDUNESP_REMOTE_DIR:-software/nn}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

mkdir -p "$ROOT_DIR/results/meeting01"
rsync -avz -e "sshpass -e ssh" \
  "${GRIDUNESP_USER}@${HOST}:${REMOTE_DIR}/results/meeting01/" \
  "$ROOT_DIR/results/meeting01/"

echo "[pull-progress] synced -- view with:"
echo "[pull-progress]   ${ROOT_DIR}/.venv/bin/python3 scripts/pipeline/meeting01/monitor.py --run-tag meeting01_loso"
echo "[pull-progress]   (or --plain / --once / --rank N -- see monitor.py --help)"
