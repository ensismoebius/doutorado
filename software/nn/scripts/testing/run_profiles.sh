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
# Clean the scratch dir and always restore the cursor (the monitor hides it).
trap 'rm -rf "$TMP"; [ -t 1 ] && printf "\033[?25h"' EXIT INT TERM
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
mkdir -p "$TMP/active"

# Stable, human-reachable per-profile logs (overwritten each run) so you can
# `tail -f results/run_logs/<profile>.log` to watch any single worker in full.
LOGDIR="results/run_logs"
mkdir -p "$LOGDIR"

# Live dashboard: with JOBS>1 the workers' own progress bars are hidden in their
# logs (concurrent bars can't share one terminal), so a monitor renders, in
# place, the latest progress line of each running worker plus a pass/fail tally.
# On a TTY it is on by default; E05_MONITOR=0 falls back to plain event lines.
MON=0
if [ "$JOBS" -gt 1 ] && [ "$tty_out" -eq 1 ] && [ "${E05_MONITOR:-1}" != 0 ]; then MON=1; fi

# Last meaningful progress line of a worker log: split CR-updated bars into
# lines, strip ANSI, drop blanks, keep the final one (truncated to terminal-ish).
mon_extract() {
    tr '\r' '\n' < "$1" 2>/dev/null | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' \
        | grep -aE '[^[:space:]]' | tail -1 | cut -c1-110
}

# Background dashboard. Redraws in place (cursor-up) every E05_MONITOR_INTERVAL
# seconds while $TMP/mon.on exists. It is the only thing printing to the TTY in
# monitor mode (worker event lines are suppressed), so the redraw stays aligned.
monitor_loop() {
    local prev=0 interval="${E05_MONITOR_INTERVAL:-2}" id nm pl p f d n k ln
    local -a lines
    printf '\033[?25l'   # hide cursor
    while [ -e "$TMP/mon.on" ]; do
        p=$(grep -c '^PASS' "$TMP/tally" 2>/dev/null); p=${p:-0}
        f=$(grep -c '^FAIL' "$TMP/tally" 2>/dev/null); f=${f:-0}
        d=$((p + f))
        lines=("$(printf '── e05 monitor ── %s ── running %d · done %d/%d · pass %d · fail %d ──' \
            "$(date +%T)" "$(ls "$TMP/active" 2>/dev/null | wc -l)" "$d" "$npending" "$p" "$f")")
        for id in $(ls "$TMP/active" 2>/dev/null | sort -n); do
            nm=$(cat "$TMP/active/$id" 2>/dev/null); [ -z "$nm" ] && continue
            pl=$(mon_extract "$LOGDIR/$nm.log")
            lines+=("$(printf '  %-38.38s %s' "$nm" "${pl:-starting…}")")
        done
        [ "$prev" -gt 0 ] && printf '\033[%dA' "$prev"
        for ln in "${lines[@]}"; do printf '\033[2K%s\n' "$ln"; done
        n=${#lines[@]}
        if [ "$n" -lt "$prev" ]; then          # active set shrank — clear stale rows
            for ((k = 0; k < prev - n; k++)); do printf '\033[2K\n'; done
            printf '\033[%dA' $((prev - n))
        fi
        prev=$n
        sleep "$interval"
    done
    printf '\033[?25h'   # restore cursor
}

# Run one profile: execute, capture failure, checkpoint, drop its active marker.
# State/tally/failure writes are serialised via flock so parallel workers never
# interleave. Output goes to the stable per-profile log. The trailing status
# line is suppressed under the monitor (the dashboard shows it instead).
run_profile() {
    local slot="$1" f="$2" name log ec sw err
    name=$(basename "$f" .json); log="$LOGDIR/$name.log"
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
    rm -f "$TMP/active/$slot"                      # free the slot (worker done)
    if [ "$MON" -eq 0 ]; then
        printf '%s  %s%s\n' "$sw" "$name" \
            "$([ "$sw" = FAIL ] && printf ' -> %s' "${err:-<non-zero exit>}")"
    fi
}

# Partition into pending vs. already-done (resume). Skips are printed up front,
# before the monitor takes over the screen.
PENDING=()
for f in "${PROFILES[@]}"; do
    if [ -n "${DONE[$f]:-}" ]; then
        skip=$((skip + 1)); printf 'SKIP (done)  %s\n' "$(basename "$f")"
    else
        PENDING+=("$f")
    fi
done
npending=${#PENDING[@]}
echo "profiles: $total  (pending $npending, skipped $skip)  from $ROOT"
echo "logs: $LOGDIR/<profile>.log   (tail -f to watch one worker in full)"
echo "state: $STATE"

if [ "$JOBS" -le 1 ]; then
    # Serial: the binary's live progress bars render directly; tee also captures
    # to the stable log. Pipe/CI → redirect (keeps the log free of ANSI noise).
    for f in "${PENDING[@]}"; do
        i=$((i + 1)); name=$(basename "$f" .json)
        now=$(date +%s); elapsed=$((now - start))
        printf '[%d/%d]  %02d:%02d  running: %s\n' \
            "$i" "$npending" $((elapsed / 60)) $((elapsed % 60)) "$name"
        if [ "$tty_out" -eq 1 ]; then
            "$BIN" --config "$f" 2>&1 | tee "$LOGDIR/$name.log"; ec=${PIPESTATUS[0]}
        else
            "$BIN" --config "$f" > "$LOGDIR/$name.log" 2>&1; ec=$?
        fi
        if [ "$ec" -eq 0 ]; then sw=PASS; else sw=FAIL; fi
        printf '%s %s\n' "$sw" "$f" >> "$STATE"
        echo "$sw" >> "$TMP/tally"
        if [ "$sw" = FAIL ]; then
            err=$(grep -aiE "Error|Exception|terminate|Assertion|what\(\)|abort" "$LOGDIR/$name.log" \
                  | sed 's/\x1b\[[0-9;]*[A-Za-z]//g; s/\r//g' | tail -1)
            { echo "FAIL: $f"; echo "   ${err:-<non-zero exit, no error line captured>}"; } >> "$TMP/failures"
            echo "FAIL [$i/$npending] $name -> ${err:-<non-zero exit>}"
        fi
    done
else
    # Parallel pool: at most JOBS workers in flight. Slot occupancy is tracked by
    # marker files in $TMP/active (the monitor is a background job too, so we
    # count markers rather than `jobs` to avoid counting it as a worker).
    [ "$MON" -eq 1 ] && { : > "$TMP/mon.on"; monitor_loop & mon_pid=$!; }
    idx=0; worker_pids=()
    for f in "${PENDING[@]}"; do
        idx=$((idx + 1)); name=$(basename "$f" .json)
        echo "$name" > "$TMP/active/$idx"
        [ "$MON" -eq 0 ] && printf 'start: %s\n' "$name"
        run_profile "$idx" "$f" &
        worker_pids+=("$!")
        while [ "$(ls "$TMP/active" 2>/dev/null | wc -l)" -ge "$JOBS" ]; do wait -n; done
    done
    # Drain workers ONLY (bare `wait` would also block on the monitor, which does
    # not stop until mon.on is removed below — a deadlock).
    [ "${#worker_pids[@]}" -gt 0 ] && wait "${worker_pids[@]}" 2>/dev/null || true
    if [ "$MON" -eq 1 ]; then rm -f "$TMP/mon.on"; wait "$mon_pid" 2>/dev/null || true; fi
fi

# grep -c prints 0 (exit 1) when the tally is empty; set -u tolerates that.
pass=$(grep -c '^PASS' "$TMP/tally" 2>/dev/null); pass=${pass:-0}
fail=$(grep -c '^FAIL' "$TMP/tally" 2>/dev/null); fail=${fail:-0}

now=$(date +%s); elapsed=$((now - start))
echo "=== summary: $pass passed, $fail failed, $skip skipped of $total  (elapsed $((elapsed / 60))m$((elapsed % 60))s) ==="
if [ $((pass + fail + skip)) -eq "$total" ] && [ "$fail" -eq 0 ]; then
    echo "all profiles complete — remove $STATE to force a full re-run next time."
fi
if [ "$fail" -gt 0 ]; then
    echo "--- failures ---"
    cat "$TMP/failures"
fi
[ "$fail" -eq 0 ]
