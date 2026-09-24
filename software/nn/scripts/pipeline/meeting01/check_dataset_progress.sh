#!/usr/bin/env bash
# check_dataset_progress.sh — one-shot status check for ensure_datasets.sh's fetch
# on GridUnesp: per-dataset file count against a known total, whether anything is
# still actively downloading/building, and — if something is — a short throughput
# sample and rough ETA for whichever dataset is currently incomplete.
#
# Usage (first run prompts for username + password and saves them to .env next to
# this script -- see _gridunesp_env.sh; later runs read .env instead of asking
# again):
#   ./scripts/pipeline/meeting01/check_dataset_progress.sh            # full check
#   ./scripts/pipeline/meeting01/check_dataset_progress.sh --no-eta   # skip the 30s sample
#
# Known totals (2026-09-24): fsdDataset 3000 .wav (Jakobovski/free-spoken-digit-dataset,
# verified against a complete local clone), audioMNIST_8k 30000 .wav (60 speakers x 500
# recordings, soerenab/AudioMNIST), eegmmidb 1526 .edf, siena 41 .edf (both counted from
# PhysioNet's own RECORDS manifest at https://physionet.org/files/<name>/1.0.0/RECORDS).
# If any of these datasets' upstream layout ever changes, update the `totals`/`exts`
# tables below to match -- a stale total would just make the percentage wrong, not
# silently misreport MISSING/present, since presence is still checked separately.
#
# One SSH session per invocation (Fail2Ban lockout, see
# .wiki/Guides/GridUnesp-Deployment.md) -- the optional ETA sample (two `du` reads 30s
# apart) runs INSIDE that one session, not as a second connection. Safe to re-run any
# time; read-only on the remote (no files are created, moved, or deleted).
set -euo pipefail

# shellcheck source=./_gridunesp_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gridunesp_env.sh"

HOST="${GRIDUNESP_HOST:-access.grid.unesp.br}"
REMOTE_DIR="${GRIDUNESP_REMOTE_DIR:-software/nn}"

DO_ETA=1
for arg in "$@"; do
  case "$arg" in
    --no-eta) DO_ETA=0 ;;
    *)
      echo "check_dataset_progress.sh: unknown argument '$arg' (only --no-eta is accepted)" >&2
      exit 1
      ;;
  esac
done

sshpass -e ssh -o ConnectTimeout=15 "${GRIDUNESP_USER}@${HOST}" \
  bash -s -- "$GRIDUNESP_USER" "$REMOTE_DIR" "$DO_ETA" <<'REMOTE'
set -euo pipefail
REMOTE_USER="$1"
REMOTE_DIR="$2"
DO_ETA="$3"
DBROOT="/home/${REMOTE_USER}/Documentos/academico/UNESP/doutorado/databases"

echo "=== active processes ==="
active="$(pgrep -af 'ensure_datasets|wget|cmake --preset|cmake --build|srun|gridunesp_deploy' || true)"
if [[ -n "$active" ]]; then
  echo "$active"
else
  echo "nothing matching running"
fi

echo
echo "=== dataset status ==="
declare -A totals=( [fsdDataset]=3000 [audioMNIST_8k]=30000 [eegmmidb]=1526 [siena]=41 )
declare -A exts=( [fsdDataset]=wav [audioMNIST_8k]=wav [eegmmidb]=edf [siena]=edf )
order=(fsdDataset audioMNIST_8k eegmmidb siena)

all_done=1
incomplete_dir=""
incomplete_ext=""
incomplete_total=0
for d in "${order[@]}"; do
  dir="$DBROOT/$d"
  ext="${exts[$d]}"
  total="${totals[$d]}"
  if [[ -d "$dir" ]]; then
    n=$(find "$dir" -name "*.${ext}" 2>/dev/null | wc -l)
    sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
    if (( n >= total )); then
      echo "$d: COMPLETE ($n/$total .$ext, $sz)"
    else
      pct=$(( n * 100 / total ))
      echo "$d: ${pct}% ($n/$total .$ext, $sz)"
      all_done=0
      if [[ -z "$incomplete_dir" ]]; then
        incomplete_dir="$dir"
        incomplete_ext="$ext"
        incomplete_total="$total"
      fi
    fi
  else
    echo "$d: MISSING (0/$total .$ext)"
    all_done=0
    if [[ -z "$incomplete_dir" ]]; then
      incomplete_dir="$dir"
      incomplete_ext="$ext"
      incomplete_total="$total"
    fi
  fi
done

echo
echo "=== build state ==="
if cd "$REMOTE_DIR" 2>/dev/null; then
  if [[ -f out/build/max-performance/CMakeCache.txt ]]; then
    echo "configured: yes"
  else
    echo "configured: no"
    all_done=0
  fi
  if [[ -f out/build/max-performance/src/experiments/meeting01/meeting01 ]]; then
    echo "binary built: yes"
  else
    echo "binary built: no"
    all_done=0
  fi
else
  echo "$REMOTE_DIR not found on remote"
  all_done=0
fi

if (( all_done )); then
  echo
  echo "=== ALL DONE: datasets complete, binary built -- ready to sbatch ==="
  exit 0
fi

if [[ "$DO_ETA" == "1" && -n "$incomplete_dir" && -n "$active" ]]; then
  echo
  echo "=== throughput sample (30s, $incomplete_dir) ==="
  s1=$(du -sb "$incomplete_dir" 2>/dev/null | cut -f1 || echo 0)
  n1=$(find "$incomplete_dir" -name "*.${incomplete_ext}" 2>/dev/null | wc -l)
  sleep 30
  s2=$(du -sb "$incomplete_dir" 2>/dev/null | cut -f1 || echo 0)
  n2=$(find "$incomplete_dir" -name "*.${incomplete_ext}" 2>/dev/null | wc -l)

  delta=$(( s2 - s1 ))
  rate=$(( delta / 30 ))
  if (( rate <= 0 || n2 == 0 )); then
    echo "rate: ~0 B/s over this 30s window -- stalled, between files, or nothing" \
         "actually running (re-check the process list above)"
  else
    avg_bytes_per_file=$(( s2 / n2 ))
    remaining_files=$(( incomplete_total - n2 ))
    remaining_bytes=$(( avg_bytes_per_file * remaining_files ))
    eta_s=$(( remaining_bytes / rate ))
    eta_h=$(( eta_s / 3600 ))
    eta_m=$(( (eta_s % 3600) / 60 ))
    echo "rate: $(( rate / 1024 )) KB/s"
    echo "ETA (rough, single 30s sample): ~${eta_h}h ${eta_m}m for $incomplete_dir"
  fi
elif [[ "$DO_ETA" == "1" && -n "$incomplete_dir" && -z "$active" ]]; then
  echo
  echo "=== no active process -- fetch/build appears stopped, not just slow ==="
fi
REMOTE
