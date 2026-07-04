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
# On a terminal, each profile shows its live bars (dataset/feature/epochs/folds).
# Usage:  ./scripts/testing/run_smoke.sh [phase00|phase01|all]   # default: all
# Binary selection (works with any CMake profile):
#   auto: most recently built out/build/*/…/experiment05
#   E05_BUILD=max-performance-opencl ./scripts/testing/run_smoke.sh
#   E05_BIN=/abs/path/to/experiment05 ./scripts/testing/run_smoke.sh
set -u

cd "$(dirname "$0")/../.." # -> software/nn

# Locate the experiment05 binary under any CMake build profile.
#   E05_BIN=/path/to/experiment05   → explicit override
#   E05_BUILD=<name>                → out/build/<name>/…  (e.g. max-performance-opencl)
#   otherwise: auto-pick the most recently built experiment05 across out/build/*
if [ -n "${E05_BIN:-}" ]; then
    BIN="$E05_BIN"
elif [ -n "${E05_BUILD:-}" ]; then
    BIN="out/build/${E05_BUILD}/src/experiments/05/experiment05"
else
    BIN=$(find out/build -maxdepth 5 -type f -name experiment05 \
              -path '*/experiments/05/experiment05' -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d' ' -f2-)
fi

if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
    echo "experiment05 binary not found."
    echo "build it in any profile, e.g.:"
    echo "  cmake --build out/build/max-performance --target experiment05 -j\$(nproc)"
    echo "or point at one:  E05_BIN=/path/to/experiment05 $0 $*"
    exit 1
fi
echo "using binary: $BIN"

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
i=0
: > "$TMP/failures"

mapfile -t PROFILES < <(find "$ROOT" -name '*.json' | sort)
total=${#PROFILES[@]}
if [ "$total" -eq 0 ]; then
    echo "no smoke profiles under $ROOT — build (regenerates mirror) or run make_smoke_profiles.py"
    exit 1
fi

start=$(date +%s)
# Show a live status line only on a terminal; fall back to per-line logging in
# pipes/CI so the output stays readable.
tty_out=0; [ -t 1 ] && tty_out=1

echo "smoke: $total profiles from $ROOT"
for f in "${PROFILES[@]}"; do
    i=$((i + 1))
    name=$(basename "$f")
    now=$(date +%s); elapsed=$((now - start))
    printf '[%d/%d] pass=%d fail=%d  %02d:%02d  running: %s\n' \
        "$i" "$total" "$pass" "$fail" $((elapsed / 60)) $((elapsed % 60)) "$name"

    # On a terminal, let the binary's live progress bars (dataset / feature
    # extraction / epochs / folds) render while tee captures output for failure
    # diagnosis. In a pipe/CI, redirect to keep the log free of ANSI bar noise.
    if [ "$tty_out" -eq 1 ]; then
        "$BIN" --config "$f" 2>&1 | tee "$TMP/out"; ec=${PIPESTATUS[0]}
    else
        "$BIN" --config "$f" > "$TMP/out" 2>&1; ec=$?
    fi

    if [ "$ec" -eq 0 ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        err=$(grep -aiE "Error|Exception|terminate|Assertion|what\(\)|abort" "$TMP/out" \
              | sed 's/\x1b\[[0-9;]*[A-Za-z]//g; s/\r//g' | tail -1)
        { echo "FAIL: $f"; echo "   ${err:-<non-zero exit, no error line captured>}"; } >> "$TMP/failures"
        echo "FAIL [$i/$total] $name -> ${err:-<non-zero exit>}"
    fi
done

now=$(date +%s); elapsed=$((now - start))
if [ "$tty_out" -eq 1 ]; then printf '\r\033[K'; fi
echo "=== smoke summary: $pass passed, $fail failed of $total  (elapsed $((elapsed / 60))m$((elapsed % 60))s) ==="
if [ "$fail" -gt 0 ]; then
    echo "--- failures ---"
    cat "$TMP/failures"
fi
[ "$fail" -eq 0 ]
