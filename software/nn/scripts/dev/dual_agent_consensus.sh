#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dev/dual_agent_consensus.sh \
    --task-id TASK_001 \
    --proposer docs/consensus/TASK_001.proposer.md \
    --antagonist docs/consensus/TASK_001.antagonist.md \
    --checks "cmake --build --preset=max-performance -j4;;ctest --test-dir out/build/max-performance -R e05 --output-on-failure"

Expected report fields in both files:
  TASK_ID: <id>
  SOLUTION_SHA256: <sha256>
  AGREE: yes|no

Behavior:
  - fails if either AGREE is not yes
  - fails if TASK_ID differs
  - fails if SOLUTION_SHA256 differs
  - runs optional checks split by ';;'
EOF
}

TASK_ID=""
PROPOSER=""
ANTAGONIST=""
CHECKS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id)
      TASK_ID="${2:-}"
      shift 2
      ;;
    --proposer)
      PROPOSER="${2:-}"
      shift 2
      ;;
    --antagonist)
      ANTAGONIST="${2:-}"
      shift 2
      ;;
    --checks)
      CHECKS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_ID" || -z "$PROPOSER" || -z "$ANTAGONIST" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 2
fi

if [[ ! -f "$PROPOSER" ]]; then
  echo "Proposer report not found: $PROPOSER" >&2
  exit 2
fi
if [[ ! -f "$ANTAGONIST" ]]; then
  echo "Antagonist report not found: $ANTAGONIST" >&2
  exit 2
fi

extract_field() {
  local key="$1"
  local file="$2"
  local value
  value=$(grep -E "^${key}:" "$file" | head -n1 | sed -E "s/^${key}:[[:space:]]*//") || true
  printf '%s' "$value"
}

P_TASK=$(extract_field "TASK_ID" "$PROPOSER")
A_TASK=$(extract_field "TASK_ID" "$ANTAGONIST")
P_SHA=$(extract_field "SOLUTION_SHA256" "$PROPOSER")
A_SHA=$(extract_field "SOLUTION_SHA256" "$ANTAGONIST")
P_AGREE=$(extract_field "AGREE" "$PROPOSER" | tr '[:upper:]' '[:lower:]')
A_AGREE=$(extract_field "AGREE" "$ANTAGONIST" | tr '[:upper:]' '[:lower:]')

if [[ -z "$P_TASK" || -z "$A_TASK" || -z "$P_SHA" || -z "$A_SHA" || -z "$P_AGREE" || -z "$A_AGREE" ]]; then
  echo "Missing required fields in one or both reports." >&2
  exit 1
fi

if [[ "$P_TASK" != "$TASK_ID" || "$A_TASK" != "$TASK_ID" ]]; then
  echo "TASK_ID mismatch. cli=$TASK_ID proposer=$P_TASK antagonist=$A_TASK" >&2
  exit 1
fi

if [[ "$P_AGREE" != "yes" || "$A_AGREE" != "yes" ]]; then
  echo "Consensus failed. proposer_agree=$P_AGREE antagonist_agree=$A_AGREE" >&2
  exit 1
fi

if [[ "$P_SHA" != "$A_SHA" ]]; then
  echo "Consensus failed. SOLUTION_SHA256 differs." >&2
  echo "proposer=$P_SHA" >&2
  echo "antagonist=$A_SHA" >&2
  exit 1
fi

echo "Consensus OK for $TASK_ID with solution $P_SHA"

if [[ -n "$CHECKS" ]]; then
  OLDIFS="$IFS"
  IFS=';;' read -r -a CMD_LIST <<< "$CHECKS"
  IFS="$OLDIFS"

  for cmd in "${CMD_LIST[@]}"; do
    trimmed=$(echo "$cmd" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
    if [[ -z "$trimmed" ]]; then
      continue
    fi
    echo "Running check: $trimmed"
    eval "$trimmed"
  done
fi

echo "Gate passed. Safe to implement/merge this solution."
