#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Run inference binary (if available) and measure latency/energy placeholders
# Usage: ./scripts/run_measurement.sh path/to/inference_binary [args]

BIN=${1:-}
shift || true
if [ -z "$BIN" ]; then
  echo "No inference binary provided. This script measures an existing binary." >&2
  echo "Usage: $0 path/to/inference_binary [args]" >&2
  exit 1
fi

if [ ! -x "$BIN" ]; then
  echo "Binary $BIN not found or not executable" >&2
  exit 1
fi

OUTDIR=measurements
mkdir -p "$OUTDIR"

echo "Running inference binary: $BIN $@"
START=$(date +%s.%N)
"$BIN" "$@"
END=$(date +%s.%N)
ELAPSED=$(python3 - <<PY
from decimal import Decimal
print(Decimal("$END")-Decimal("$START"))
PY
)

echo "elapsed_seconds=$ELAPSED" > "$OUTDIR/last_run.txt"
echo "Note: energy measurement requires external meter. Use measured event counts or integrate power meter for energy." >> "$OUTDIR/last_run.txt"
echo "Results written to $OUTDIR/last_run.txt"
