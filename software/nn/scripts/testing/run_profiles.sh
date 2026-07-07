#!/usr/bin/env bash
#
# Run the REAL Experiment05 profiles (full parameters — this is the actual
# experiment, not the smoke mirror). Shows live progress and captures failures.
#
# Pipeline order:
#   1) run_profiles.sh phase00     → paraconsistent ranking CSVs in results/phase00
#   2) e05_phase00_rank.py         → winners.json
#   3) e05_apply_winner.py         → injects the winner into the phase01 profiles
#   4) run_profiles.sh phase01     → DSNN authentication, EER/AUC in results/phase01
#
# Requires: experiment05 built, and the dataset (dataset.root) present.
# HEAVY: phase00 = 282 profiles, phase01 = 32, each with experiment.repeats runs.
# Run in the background / overnight.
#
# On a terminal, each profile shows its live bars (dataset/feature/epochs/folds).
#
# Crash/power-loss safe: each completed profile is checkpointed to
# results/run_profiles_<scope>.state. On restart an existing state prompts
# resume (skip completed) vs. start over. Non-interactive default = resume.
#   RESUME=1 → force resume    FRESH=1 → force start over
#
# Usage:  ./scripts/testing/run_profiles.sh [phase00|phase01|all]   # default: all
# Binary selection (any CMake profile):
#   auto: most recently built out/build/*/…/experiment05
#   E05_BUILD=max-performance ./scripts/testing/run_profiles.sh phase00
#   E05_BIN=/abs/path/to/experiment05 ./scripts/testing/run_profiles.sh
set -u

cd "$(dirname "$0")/../.." # -> software/nn

# Locate the experiment05 binary under any CMake build profile.
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
    echo "experiment05 binary not found. Build it, e.g.:"
    echo "  cmake --build out/build/max-performance --target experiment05 -j\$(nproc)"
    echo "or point at one:  E05_BIN=/path/to/experiment05 $0 $*"
    exit 1
fi
echo "using binary: $BIN"

# ── Parallelism (memory-gated) ───────────────────────────────────────────────
# Run several profiles at once when RAM allows. Each profile is an independent
# run (own result file by run_tag), so phase00/phase01 parallelise safely.
# Job count = min(free_RAM / per-job, nproc), capped at 4 to avoid GPU
# oversubscription. Override with E05_JOBS; tune per-job budget with
# E05_JOB_MEM_MB (default 2048). JOBS=1 keeps the live progress-bar UX.
avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
[ -z "${avail_mb:-}" ] && avail_mb=2048
per_mb=${E05_JOB_MEM_MB:-2048}
jobs_mem=$(( avail_mb / per_mb )); [ "$jobs_mem" -lt 1 ] && jobs_mem=1
cpus=$(nproc)
if [ -n "${E05_JOBS:-}" ]; then
    JOBS=$E05_JOBS
else
    JOBS=$(( jobs_mem < cpus ? jobs_mem : cpus ))
    [ "$JOBS" -gt 4 ] && JOBS=4
fi
[ "$JOBS" -lt 1 ] && JOBS=1
echo "parallelism: JOBS=$JOBS  (avail ${avail_mb}MB / ${per_mb}MB per job, ${cpus} cpus)"

SCOPE="${1:-all}"
case "$SCOPE" in
    phase00) ROOT="src/experiments/05/profiles/phase00" ;;
    phase01) ROOT="src/experiments/05/profiles/phase01" ;;
    all)     ROOT="src/experiments/05/profiles" ;;   # phase00 + phase01 + debug.json
    *) echo "usage: $0 [phase00|phase01|all]"; exit 1 ;;
esac

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0
fail=0
skip=0
i=0
: > "$TMP/failures"

# Exclude the smoke mirror; run only real profiles.
mapfile -t PROFILES < <(find "$ROOT" -name '*.json' -not -path '*/smoke/*' | sort)
total=${#PROFILES[@]}
if [ "$total" -eq 0 ]; then
    echo "no profiles under $ROOT"
    exit 1
fi

# ── Checkpoint / resume ──────────────────────────────────────────────────────
# Each completed profile is appended to a persistent state file (survives a
# power loss — the binary's per-profile result files are already on disk, so
# this is just the skip-list). A profile interrupted mid-run is NOT recorded and
# re-runs. On start, an existing state prompts resume vs. start-over.
#   RESUME=1  → force resume (skip prompt)   FRESH=1 → force start-over
mkdir -p results
STATE="results/run_profiles_${SCOPE}.state"
declare -A DONE
if [ -s "$STATE" ]; then
    done_n=$(wc -l < "$STATE")
    echo "Found a previous run: $done_n profile(s) already completed → $STATE"
    if [ -n "${FRESH:-}" ]; then ans="S"
    elif [ -n "${RESUME:-}" ]; then ans="R"
    elif [ -t 0 ]; then
        read -r -p "Resume (skip completed) [R], start over [S], abort [A]? " ans
    else
        ans="R"; echo "non-interactive → resuming (set FRESH=1 to start over)"
    fi
    case "${ans:-A}" in
        R|r) while read -r _st path; do DONE["$path"]=1; done < "$STATE"
             echo "resuming: ${#DONE[@]} profile(s) will be skipped" ;;
        S|s) : > "$STATE"; echo "starting over (state cleared)" ;;
        *)   echo "aborted"; exit 1 ;;
    esac
fi

start=$(date +%s)
tty_out=0; [ -t 1 ] && tty_out=1
: > "$TMP/tally"      # one PASS/FAIL line per completed profile (parallel-safe)

# Run one profile: execute, capture failure, checkpoint. State/tally/failure
# writes are serialised via flock so parallel workers never interleave. The
# per-profile output goes to its own log ($1 index) — no shared "out" file.
run_profile() {
    local i="$1" f="$2" name log ec sw err
    name=$(basename "$f"); log="$TMP/log.$i"
    "$BIN" --config "$f" > "$log" 2>&1; ec=$?
    if [ "$ec" -eq 0 ]; then sw=PASS; else sw=FAIL; fi
    (
        flock 9
        printf '%s %s\n' "$sw" "$f" >> "$STATE"   # resume checkpoint
        echo "$sw" >> "$TMP/tally"
        if [ "$sw" = FAIL ]; then
            err=$(grep -aiE "Error|Exception|terminate|Assertion|what\(\)|abort" "$log" \
                  | sed 's/\x1b\[[0-9;]*[A-Za-z]//g; s/\r//g' | tail -1)
            { echo "FAIL: $f"; echo "   ${err:-<non-zero exit, no error line captured>}"; } >> "$TMP/failures"
        fi
    ) 9>"$TMP/lock"
    printf '[%d/%d] %s  %s%s\n' "$i" "$total" "$sw" "$name" \
        "$([ "$sw" = FAIL ] && printf ' -> %s' "${err:-<non-zero exit>}")"
}

echo "profiles: $total from $ROOT  (state: $STATE)"
for f in "${PROFILES[@]}"; do
    i=$((i + 1))
    name=$(basename "$f")

    # Skip profiles already completed in a previous run (resume).
    if [ -n "${DONE[$f]:-}" ]; then
        skip=$((skip + 1))
        printf '[%d/%d] SKIP (done)  %s\n' "$i" "$total" "$name"
        continue
    fi

    if [ "$JOBS" -le 1 ]; then
        # Serial: live progress bars (dataset / feature / epochs / folds) render
        # while tee captures output for failure diagnosis. Pipe/CI → redirect.
        now=$(date +%s); elapsed=$((now - start))
        printf '[%d/%d] skip=%d  %02d:%02d  running: %s\n' \
            "$i" "$total" "$skip" $((elapsed / 60)) $((elapsed % 60)) "$name"
        if [ "$tty_out" -eq 1 ]; then
            "$BIN" --config "$f" 2>&1 | tee "$TMP/log.$i"; ec=${PIPESTATUS[0]}
        else
            "$BIN" --config "$f" > "$TMP/log.$i" 2>&1; ec=$?
        fi
        if [ "$ec" -eq 0 ]; then sw=PASS; else sw=FAIL; fi
        printf '%s %s\n' "$sw" "$f" >> "$STATE"
        echo "$sw" >> "$TMP/tally"
        if [ "$sw" = FAIL ]; then
            err=$(grep -aiE "Error|Exception|terminate|Assertion|what\(\)|abort" "$TMP/log.$i" \
                  | sed 's/\x1b\[[0-9;]*[A-Za-z]//g; s/\r//g' | tail -1)
            { echo "FAIL: $f"; echo "   ${err:-<non-zero exit, no error line captured>}"; } >> "$TMP/failures"
            echo "FAIL [$i/$total] $name -> ${err:-<non-zero exit>}"
        fi
    else
        # Parallel pool: keep at most JOBS workers in flight. wait -n returns as
        # soon as any worker exits, freeing a slot for the next profile.
        printf '[%d/%d] start: %s\n' "$i" "$total" "$name"
        run_profile "$i" "$f" &
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    fi
done
wait   # drain remaining parallel workers

# grep -c prints 0 (exit 1) when the tally is empty; set -u tolerates that.
pass=$(grep -c '^PASS' "$TMP/tally" 2>/dev/null); pass=${pass:-0}
fail=$(grep -c '^FAIL' "$TMP/tally" 2>/dev/null); fail=${fail:-0}

now=$(date +%s); elapsed=$((now - start))
if [ "$tty_out" -eq 1 ]; then printf '\r\033[K'; fi
echo "=== summary: $pass passed, $fail failed, $skip skipped of $total  (elapsed $((elapsed / 60))m$((elapsed % 60))s) ==="
if [ $((pass + fail + skip)) -eq "$total" ] && [ "$fail" -eq 0 ]; then
    echo "all profiles complete — remove $STATE to force a full re-run next time."
fi
if [ "$fail" -gt 0 ]; then
    echo "--- failures ---"
    cat "$TMP/failures"
fi
[ "$fail" -eq 0 ]
