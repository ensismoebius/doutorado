#!/usr/bin/env bash
#
# Run the REAL Experiment05 profiles (full parameters — this is the actual
# experiment, not the smoke mirror). Shows live progress and captures failures.
#
# Pipeline order:
#   1) run_e05_profiles.sh phase00              → paraconsistent ranking CSVs in results/phase00
#   2) pipeline/e05/01_e05_phase00_rank.py      → winners.json
#   3) pipeline/e05/02_e05_apply_winner.py      → injects the winner into the phase01 profiles
#   4) run_e05_profiles.sh phase01              → DSNN authentication, EER/AUC in results/phase01
#
# Requires: experiment05 built, and the dataset (dataset.root) present.
# HEAVY: phase00 = 208 profiles, phase01 = 32, each with experiment.repeats runs.
# Run in the background / overnight.
#
# On a terminal, each profile shows its live bars (dataset/feature/epochs/folds).
#
# Crash/power-loss safe: each completed profile is checkpointed to
# results/run_profiles_<scope>.state. On restart an existing state prompts
# resume (skip completed) vs. start over. Non-interactive default = resume.
#   RESUME=1 → force resume    FRESH=1 → force start over
#
# Usage:  ./scripts/testing/run_e05_profiles.sh [phase00|phase01|all]
#         No argument on a terminal → interactive menu. No argument in a
#         pipe/CI → defaults to `all`.
# Binary selection (any CMake profile):
#   auto: most recently built out/build/*/…/experiment05
#   E05_BUILD=max-performance ./scripts/testing/run_e05_profiles.sh phase00
#   E05_BIN=/abs/path/to/experiment05 ./scripts/testing/run_e05_profiles.sh
set -u

cd "$(dirname "$0")/../.." # -> software/nn

# Locate the experiment05 binary under any CMake build profile.
# Auto-pick prefers the CPU (max-performance) build when it exists. This is the reference
# backend for the thesis — the same one the Guayaquil paper pipeline defaults to
# (01_e04_run_article_profiles.sh), so both experiments report from one backend. It is also
# the right default on the merits: these profiles' networks are tiny (kernel-launch-bound on
# GPU), and "most recently built" used to silently switch runs to whatever was rebuilt last.
# Override with E05_BUILD/E05_BIN to target another backend.
if [ -n "${E05_BIN:-}" ]; then
    BIN="$E05_BIN"
elif [ -n "${E05_BUILD:-}" ]; then
    BIN="out/build/${E05_BUILD}/src/experiments/05/experiment05"
elif [ -x "out/build/max-performance/src/experiments/05/experiment05" ]; then
    BIN="out/build/max-performance/src/experiments/05/experiment05"
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
# E05_JOB_MEM_MB (default 5120 — measured snn-ae/poisson voice profiles peak
# ~4.4GB RSS+swap each, EEG profiles ~2.1GB; the old 2048 default let 4 voice
# jobs oversubscribe a 17GB box into heavy swap). JOBS=1 keeps the live
# progress-bar UX.
avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
[ -z "${avail_mb:-}" ] && avail_mb=2048
per_mb=${E05_JOB_MEM_MB:-5120}
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
auto_jobs=$JOBS   # computed default, offered as the reset value in the menu

# ── What to run ──────────────────────────────────────────────────────────────
# Scope may be passed as $1 (phase00|phase01|all) for automation/CI. With no
# argument on a terminal, an interactive menu asks instead; in a pipe/CI with
# no argument it defaults to `all` (unchanged non-interactive behaviour).
SCOPE="${1:-}"
if [ -z "$SCOPE" ]; then
    if [ -t 0 ] && [ -t 1 ]; then
        while :; do
            cat <<MENU

=== Experiment05 profile runner ===
binary      : $BIN
parallelism : JOBS=$JOBS  (${cpus} cpus, ${avail_mb}MB free / ${per_mb}MB per job)

What do you want to do?
  1) phase00 — feature construction + paraconsistent ranking  (208 profiles)
  2) phase01 — DSNN authentication                            (32 profiles)
  3) all     — phase00 + phase01 + debug
  j) set parallel job count (currently $JOBS; auto-detected default $auto_jobs)
  b) (re)build the experiment05 binary first
  q) quit
MENU
            read -r -p "choice [1/2/3/j/b/q]: " ans
            case "$ans" in
                1) SCOPE=phase00; break ;;
                2) SCOPE=phase01; break ;;
                3) SCOPE=all;     break ;;
                j|J)
                    read -r -p "parallel jobs [Enter = auto ($auto_jobs)]: " njobs
                    if [ -z "$njobs" ]; then
                        JOBS=$auto_jobs
                    elif [[ "$njobs" =~ ^[0-9]+$ ]] && [ "$njobs" -ge 1 ]; then
                        JOBS=$njobs
                    else
                        echo "not a positive integer — ignored"
                    fi
                    echo "parallelism now: JOBS=$JOBS" ;;
                b|B)
                    echo "building experiment05 …"
                    if cmake --build out/build/max-performance --target experiment05 -j"$(nproc)"; then
                        echo "build ok"
                    else
                        echo "build FAILED — fix errors, then choose again"
                    fi ;;
                q|Q) echo "aborted"; exit 0 ;;
                *)   echo "pick 1, 2, 3, j, b, or q" ;;
            esac
        done
    else
        SCOPE=all   # non-interactive default
    fi
fi

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

# The binary renders TWO stacked progress bars at once — a label line ("Autoencoder
# training" / "Batches done") followed by a data line
# "  <ascii bar> │ pct%  count │ status/loss │ ETA: ..." (│ = U+2502). Grab the
# last two label+data PAIRS (4 lines) from the most recent refresh.
mon_extract() {
    tr '\r' '\n' < "$1" 2>/dev/null | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' \
        | grep -aE '[^[:space:]]' | tail -4
}

# Fixed-width (10-char) mini bar redrawn from a parsed percentage — NOT the
# binary's own ~44-char ascii bar. A fixed small width means the bar itself
# can never be what pushes the trailing ETA/loss value past the terminal-width
# clamp; only its own 10 chars are at stake, and it stays readable at any size.
mon_mini_bar() {
    local pct="$1" width=10 filled ii bar="" fill_color
    filled=$(printf '%.0f' "$pct" 2>/dev/null); : "${filled:=0}"
    [ "$filled" -lt 0 ] 2>/dev/null && filled=0
    [ "$filled" -gt 100 ] 2>/dev/null && filled=100
    fill_color=$([ "$filled" -ge 100 ] && printf '\033[36m' || printf '\033[32m') # cyan at 100%, else green
    filled=$((filled * width / 100))
    bar="$fill_color"
    for ((ii = 0; ii < width; ii++)); do
        [ "$ii" -eq "$filled" ] && bar+='\033[90m'
        if [ "$ii" -lt "$filled" ]; then bar+="#"; else bar+="-"; fi
    done
    bar+='\033[0m'
    printf '%b' "$bar"
}

# Drop the binary's own ascii bar graphic (everything up to the first │) but
# keep everything after it — pct/count, loss/status, ETA — compacted into
# "a | b | c", then prefix a small redrawn bar ([##--------]) parsed from the
# percentage in that text, so the row stays both visual and never truncates
# the actually useful trailing ETA/loss value regardless of terminal width.
mon_compact_bar() {
    local s="$1" pct=""
    if [[ "$s" == *"│"* ]]; then
        s="${s#*│}"          # drop the ascii bar graphic (first field)
        s="${s//│/|}"        # remaining separators -> plain "|"
    fi
    s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"  # trim
    s=$(printf '%s' "$s" | sed -E 's/[[:space:]]{2,}/ /g')          # collapse runs
    if [[ "$s" =~ ([0-9]+\.?[0-9]*)% ]]; then
        pct="${BASH_REMATCH[1]}"
        printf '\033[90m[\033[0m%s\033[90m]\033[0m \033[1m%s\033[0m' "$(mon_mini_bar "$pct")" "$s"
    else
        printf '%s' "$s"
    fi
}

# Background dashboard. Redraws in place (cursor-up) every E05_MONITOR_INTERVAL
# seconds while $TMP/mon.on exists. It is the only thing printing to the TTY in
# monitor mode (worker event lines are suppressed), so the redraw stays aligned.
#
# Every rendered line is clamped to the terminal width. Without this, a line
# longer than the terminal wraps onto a second physical row, which the
# cursor-up math below doesn't account for (it assumes one row per logical
# line) — the redraw then lands one row too high each frame, so old frames
# are never overwritten and the screen appears to scroll continuously.
fmt_mmss() {
    local s="$1"; printf '%02d:%02d' $((s / 60)) $((s % 60))
}

# Truncate to N *visible* columns, passing ANSI escape sequences through intact
# (never cutting mid-escape — that would leak a color state into everything
# printed after it) and always closing with a reset. Use this instead of
# "${s:0:cols}" on any line that may contain color codes.
vis_trunc() {
    local s="$1" w="$2" out="" i=0 n ch esc printable=0
    n=${#s} # separate statement: "local n=${#s}" on the same line as
            # "local s=..." reads s's OLD value, not the one just assigned
    while [ "$i" -lt "$n" ] && [ "$printable" -lt "$w" ]; do
        ch="${s:$i:1}"
        if [ "$ch" = $'\033' ]; then
            esc="$ch"; i=$((i + 1))
            if [ "${s:$i:1}" = "[" ]; then
                esc+="["; i=$((i + 1))
                while [ "$i" -lt "$n" ]; do
                    ch="${s:$i:1}"; esc+="$ch"; i=$((i + 1))
                    [[ "$ch" =~ [A-Za-z] ]] && break
                done
            fi
            out+="$esc"
        else
            out+="$ch"; printable=$((printable + 1)); i=$((i + 1))
        fi
    done
    printf '%s\033[0m' "$out"
}

monitor_loop() {
    local prev=0 interval="${E05_MONITOR_INTERVAL:-2}" id nm st now_e p f d n k ln cols
    local overall_eta worker_eta first_row k2 lbl dat row
    local -a lines pls
    printf '\033[?25l'   # hide cursor
    while [ -e "$TMP/mon.on" ]; do
        cols=$(tput cols 2>/dev/null); [ -z "$cols" ] && cols=80
        now_e=$(date +%s)
        p=$(grep -c '^PASS' "$TMP/tally" 2>/dev/null); p=${p:-0}
        f=$(grep -c '^FAIL' "$TMP/tally" 2>/dev/null); f=${f:-0}
        d=$((p + f))

        # Overall ETA: work-weighted (weights_done accumulates each finished profile's cost),
        # so a fast/slow mix doesn't lurch the estimate the way a plain profile count would.
        # Wall-clock per unit of DONE work already reflects the JOBS-way concurrency (elapsed
        # grows slower than serial while done_w grows the same), so no separate /JOBS factor is
        # needed. The monitor refreshes each frame, so a running linear extrapolation over the
        # measured rate is enough here (the serial path EMA-smooths; a live redraw doesn't need
        # to). done_w is read from the file the workers append to under flock.
        done_w=$(awk '{s+=$1} END{print s+0}' "$TMP/weights_done" 2>/dev/null); done_w=${done_w:-0}
        if [ "$done_w" -gt 0 ]; then
            overall_eta="$(fmt_hms $(( (now_e - start) * (total_weight - done_w) / done_w )))"
        else
            overall_eta="calculating"
        fi

        # Two short header lines instead of one long one: a single wide line
        # risks the terminal-width clamp below cutting it off before the ETA
        # (the most useful field) — each line here is short enough on its own
        # to never need truncating.
        lines=("$(printf '\033[1;36m── e05 monitor ──\033[0m %s ── running %d · done \033[32m%d\033[0m/%d · pass \033[32m%d\033[0m · fail \033[31m%d\033[0m ──' \
            "$(date +%T)" "$(ls "$TMP/active" 2>/dev/null | wc -l)" "$d" "$npending" "$p" "$f")")
        lines+=("$(printf '    elapsed %s   overall ETA ~ \033[35m%s\033[0m' \
            "$(fmt_hms $((now_e - start)))" "$overall_eta")")
        lines[0]="$(vis_trunc "${lines[0]}" "$cols")"; lines[1]="$(vis_trunc "${lines[1]}" "$cols")"

        for id in $(ls "$TMP/active" 2>/dev/null | sort -n); do
            nm=$(sed -n '1p' "$TMP/active/$id" 2>/dev/null); [ -z "$nm" ] && continue
            st=$(sed -n '2p' "$TMP/active/$id" 2>/dev/null); [ -z "$st" ] && st=$now_e
            worker_eta="run $(fmt_mmss $((now_e - st)))"
            mapfile -t pls < <(mon_extract "$LOGDIR/$nm.log")
            # pls holds up to 4 raw lines: [outer-label, outer-data, inner-label,
            # inner-data]. Pair them (label: compacted-data) and render one row
            # per pair — up to 2 rows/worker (epoch bar, batch bar).
            first_row=1
            for ((k2 = 0; k2 < ${#pls[@]}; k2 += 2)); do
                lbl="${pls[k2]}"
                lbl="${lbl#"${lbl%%[![:space:]]*}"}"; lbl="${lbl%"${lbl##*[![:space:]]}"}"
                dat="${pls[k2 + 1]:-}"
                if [ -n "$dat" ]; then
                    row="$lbl: $(mon_compact_bar "$dat")"
                else
                    row="$lbl"
                fi
                # Pad plain (uncolored) fields to fixed width first, then wrap in
                # color — printf's %-Ns width counts raw bytes, so coloring before
                # padding would count escape bytes against the column budget.
                if [ "$first_row" -eq 1 ]; then
                    ln=$(printf '  \033[1;36m%-30.30s\033[0m \033[35m%-9s\033[0m %s' \
                        "$nm" "$worker_eta" "$row")
                    first_row=0
                else
                    ln=$(printf '  %-30.30s %-9s %s' "" "" "$row")
                fi
                lines+=("$(vis_trunc "$ln" "$cols")")
            done
            [ "${#pls[@]}" -eq 0 ] && lines+=("$(printf '  \033[1;36m%-30.30s\033[0m \033[35m%-9s\033[0m starting…' "$nm" "$worker_eta")")
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
        profile_weight "$f" >> "$TMP/weights_done" # work-weighted ETA (monitor sums this)
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

# Work-weighted overall ETA (scripts/lib/run_eta.sh, shared with the Guayaquil runner).
# E05 mixes fast handcrafted extraction (trains nothing) with slow autoencoders / DSNN
# training. Counting profiles equally makes the ETA lurch every time the mix shifts; instead
# we weight each profile by rough cost and track seconds-per-unit-work. p00_ae_* sort before
# p00_hc_*, so the heavy kind is measured early and the rate is trustworthy for the long
# handcrafted tail. Phase01 is all DSNN (heavy) — the catch-all covers it.
source scripts/lib/run_eta.sh
profile_weight() { case "$(basename "$1")" in p00_hc_*) echo 1 ;; *) echo 20 ;; esac; }
total_weight=0; for _p in "${PENDING[@]}"; do total_weight=$((total_weight + $(profile_weight "$_p"))); done
: > "$TMP/weights_done"   # workers append each completed profile's weight; monitor sums it
eta_reset

if [ "$JOBS" -le 1 ]; then
    # Serial: the binary's live progress bars render directly; tee also captures
    # to the stable log. Pipe/CI → redirect (keeps the log free of ANSI noise).
    done_w=0
    for f in "${PENDING[@]}"; do
        i=$((i + 1)); name=$(basename "$f" .json)
        elapsed=$(( $(date +%s) - start ))
        rem_s=$(eta_remaining $(( total_weight - done_w )))
        eta=$([ -n "$rem_s" ] && printf 'eta~%s' "$(fmt_hms "$rem_s")" || echo "eta~calculating")
        printf '[%d/%d]  elapsed %s  %s  running: %s\n' \
            "$i" "$npending" "$(fmt_hms "$elapsed")" "$eta" "$name"
        # Persistent top banner inside the profile's own TUI (see experiment05 / E05_OVERALL),
        # mirroring the Guayaquil runner: the per-process bars can't know the whole-run status,
        # so the runner computes it here and hands it to the binary the same way E04 does.
        export E05_OVERALL="$(printf 'Overall  [%d/%d]  elapsed %s  ETA %s   (%s)' \
            "$i" "$npending" "$(fmt_hms "$elapsed")" "$eta" "$name")"
        p_start=$(date +%s)
        if [ "$tty_out" -eq 1 ]; then
            "$BIN" --config "$f" 2>&1 | tee "$LOGDIR/$name.log"; ec=${PIPESTATUS[0]}
        else
            "$BIN" --config "$f" > "$LOGDIR/$name.log" 2>&1; ec=$?
        fi
        eta_update "$(profile_weight "$f")" "$(( $(date +%s) - p_start ))"
        done_w=$((done_w + $(profile_weight "$f")))
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
    unset E05_OVERALL
else
    # Parallel pool: at most JOBS workers in flight. Slot occupancy is tracked by
    # marker files in $TMP/active (the monitor is a background job too, so we
    # count markers rather than `jobs` to avoid counting it as a worker).
    [ "$MON" -eq 1 ] && { : > "$TMP/mon.on"; monitor_loop & mon_pid=$!; }
    idx=0; worker_pids=()
    # Throttle on live worker PIDs, not on $TMP/active marker files: a worker
    # stopped by SIGTTOU/SIGTTIN (script backgrounded while something touches
    # the TTY) never removes its marker, and a signal-interrupted `wait -n`
    # returns without reaping — the marker-count loop then spun straight
    # through and launched EVERY pending profile at once (observed twice: ~30
    # concurrent experiment05 processes, OOM-killing unrelated work).
    prune_workers() {
        local alive=() p
        for p in "${worker_pids[@]}"; do
            kill -0 "$p" 2>/dev/null && alive+=("$p")
        done
        worker_pids=("${alive[@]}")
    }
    for f in "${PENDING[@]}"; do
        idx=$((idx + 1)); name=$(basename "$f" .json)
        printf '%s\n%s\n' "$name" "$(date +%s)" > "$TMP/active/$idx"
        [ "$MON" -eq 0 ] && printf 'start: %s\n' "$name"
        run_profile "$idx" "$f" &
        worker_pids+=("$!")
        prune_workers
        while [ "${#worker_pids[@]}" -ge "$JOBS" ]; do
            wait -n 2>/dev/null || sleep 1
            prune_workers
        done
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
