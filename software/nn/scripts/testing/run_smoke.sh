#!/usr/bin/env bash
#
# Smoke-run every Experiment05 profile under profiles/smoke/ to surface runtime
# errors that compilation cannot catch. Each profile uses tiny run parameters
# (small sample cap, 2 epochs, 2 folds) but keeps its real code-path selectors.
#
# Requires: the experiment05 binary built, and the dataset (dataset.root) present.
# Long-running (~315 runs). Regenerate the smoke profiles with
#   .venv/bin/python scripts/testing/make_smoke_profiles.py
# after editing any real profile.
#
# Usage:  ./scripts/testing/run_smoke.sh [phase00|phase01|all]   # default: all
set -u

cd "$(dirname "$0")/../.." # -> software/nn
BIN=out/build/max-performance/src/experiments/05/experiment05
if [ ! -x "$BIN" ]; then
    echo "build first: cmake --build out/build/max-performance --target experiment05 -j\$(nproc)"
    exit 1
fi

SCOPE="${1:-all}"
case "$SCOPE" in
    phase00) ROOT="src/experiments/05/profiles/smoke/phase00" ;;
    phase01) ROOT="src/experiments/05/profiles/smoke/phase01" ;;
    all)     ROOT="src/experiments/05/profiles/smoke" ;;
    *) echo "usage: $0 [phase00|phase01|all]"; exit 1 ;;
esac

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0
fail=0
: > "$TMP/failures"

while IFS= read -r f; do
    if "$BIN" --config "$f" > "$TMP/out" 2>&1; then
        pass=$((pass + 1)); printf '.'
    else
        fail=$((fail + 1)); printf 'F'
        err=$(grep -aiE "Error|Exception|terminate|Assertion|what\(\)|abort" "$TMP/out" \
              | sed 's/\x1b\[[0-9;]*[A-Za-z]//g; s/\r//g' | tail -1)
        { echo "FAIL: $f"; echo "   ${err:-<non-zero exit, no error line captured>}"; } >> "$TMP/failures"
    fi
done < <(find "$ROOT" -name '*.json' | sort)

echo ""
echo "=== smoke summary: $pass passed, $fail failed ==="
if [ "$fail" -gt 0 ]; then
    echo "--- failures ---"
    cat "$TMP/failures"
fi
[ "$fail" -eq 0 ]
