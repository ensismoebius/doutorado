#!/usr/bin/env bash
# _gridunesp_env.sh — shared GridUnesp SSH credential loading. NOT meant to be run
# directly: sourced by gridunesp_deploy.sh, remote_monitor.sh, and pull_progress.sh.
#
# First run: prompts for username (visible) and password (hidden, like a normal
# password prompt), then saves both to .env next to this script. Every later run:
# loads .env instead of prompting again -- ".env" is listed in ../../../.gitignore,
# so it is never committed.
#
# Trade-off, stated plainly: this keeps your GridUnesp password in a plaintext file
# on disk. It is written chmod 600 (owner read/write only), which stops other local
# accounts on this machine from reading it, but that is the only protection —
# treat this checkout's .env the way you would any other place holding a live
# password (do not copy it elsewhere, do not attach it to a bug report or tarball
# of this repo, etc.).
#
# Needs `sshpass` on the LOCAL machine (not the remote) to hand that password to
# ssh/rsync non-interactively: `sudo apt install sshpass` (Debian/Ubuntu) or
# `conda install -c conda-forge sshpass`.
#
# Exports on success: GRIDUNESP_USER, GRIDUNESP_PASSWORD, SSHPASS (same value as
# GRIDUNESP_PASSWORD, what `sshpass -e` reads). Callers invoke ssh/rsync through
# `sshpass -e ssh ...` / `rsync -e 'sshpass -e ssh' ...`.

_gridunesp_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_gridunesp_env_file="${_gridunesp_env_dir}/.env"

if [[ -f "$_gridunesp_env_file" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$_gridunesp_env_file"
  set +a
fi

if [[ -z "${GRIDUNESP_USER:-}" || -z "${GRIDUNESP_PASSWORD:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "$(basename "$0"): GridUnesp credentials missing (no $_gridunesp_env_file," \
         "and stdin is not a terminal to prompt for them)" >&2
    exit 1
  fi
  [[ -n "${GRIDUNESP_USER:-}" ]] || read -rp "GridUnesp username: " GRIDUNESP_USER
  [[ -n "${GRIDUNESP_USER:-}" ]] || { echo "$(basename "$0"): username cannot be empty" >&2; exit 1; }
  if [[ -z "${GRIDUNESP_PASSWORD:-}" ]]; then
    read -rsp "GridUnesp password: " GRIDUNESP_PASSWORD
    echo
  fi
  [[ -n "$GRIDUNESP_PASSWORD" ]] || { echo "$(basename "$0"): password cannot be empty" >&2; exit 1; }

  ( umask 077
    cat > "$_gridunesp_env_file" <<EOF
GRIDUNESP_USER=${GRIDUNESP_USER}
GRIDUNESP_PASSWORD=${GRIDUNESP_PASSWORD}
EOF
  )
  echo "[gridunesp] saved credentials to ${_gridunesp_env_file} (chmod 600, git-ignored) -- later runs will not ask again"
fi

command -v sshpass >/dev/null 2>&1 || {
  echo "$(basename "$0"): 'sshpass' not found on PATH -- needed to use the saved" \
       "password non-interactively. Install it: 'sudo apt install sshpass'" \
       "(Debian/Ubuntu) or 'conda install -c conda-forge sshpass'." >&2
  exit 1
}

export GRIDUNESP_USER GRIDUNESP_PASSWORD
export SSHPASS="$GRIDUNESP_PASSWORD"
