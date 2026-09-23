#!/usr/bin/env bash
# remote_monitor.sh — live meeting01 progress on GridUnesp, no local sync needed.
#
# Runs monitor.py directly on the login node over one SSH session. --plain mode
# (what this always passes) needs only the Python stdlib -- every `rich` import in
# monitor.py is function-local, used only by the default TTY dashboard -- so this
# works without activating the conda env or having `rich` installed anywhere.
#
# Usage:
#   GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh
#   GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh --once
#   GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh --rank 3
#   GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh --interval 30
#
# Extra arguments are forwarded verbatim to monitor.py (see its own --help).
#
# One SSH session per invocation, on purpose: GridUnesp's login node Fail2Ban-locks
# repeated rapid connections for 15 minutes, and reconnecting during the lockout
# restarts the timer (.wiki/Guides/GridUnesp-Deployment.md). For a continuously
# refreshing view, let monitor.py's own --interval loop run INSIDE this one SSH
# session (the default here, no --once) rather than wrapping this script in a shell
# polling loop of your own.
set -euo pipefail

: "${GRIDUNESP_USER:?set GRIDUNESP_USER=<your grid username>}"
HOST="${GRIDUNESP_HOST:-access.grid.unesp.br}"
REMOTE_DIR="${GRIDUNESP_REMOTE_DIR:-software/nn}"

exec ssh "${GRIDUNESP_USER}@${HOST}" \
  "cd ${REMOTE_DIR} && python3 scripts/pipeline/meeting01/monitor.py --plain $*"
