#!/usr/bin/env bash
#
# Run the REAL Experiment05 profiles (full parameters — this is the actual
# experiment, not the smoke mirror). Shows live progress and captures failures.
#
# Pipeline order:
#   1) run_thesis_profiles.sh phase00   → paraconsistent ranking CSVs in results/thesis/phase00,
#                                      then AUTOMATICALLY (see "Phase 00 post-processing"):
#                                        01_thesis_phase00_rank.py  → winners.json
#                                        02_thesis_apply_winner.py  → injects the winner into
#                                                                  the 32 phase01 profiles
#                                        thesis_build_phase00_paraconsistent_tables.py
#                                                                → regenerates the thesis tables
#   2) run_thesis_profiles.sh phase01   → DSNN authentication, EER/AUC in results/thesis/phase01
#
# Step 1's post-processing REWRITES tracked files under
# src/experiments/thesis/profiles/phase01/ — review that diff before committing.
# It is skipped when any profile failed (the ranking would be computed over an
# incomplete set); THESIS_FORCE_POST=1 overrides, THESIS_SKIP_POST=1 disables it entirely.
#
# Requires: thesis built, and the dataset (dataset.root) present.
# HEAVY: phase00 = 208 profiles, phase01 = 32, each with experiment.repeats runs.
# Run in the background / overnight.
#
# On a terminal, each profile shows its live bars (dataset/feature/epochs/folds).
#
# Crash/power-loss safe: each completed profile is checkpointed to
# results/thesis/run_profiles_<scope>.state. On restart an existing state prompts
# resume (skip completed) vs. start over. Non-interactive default = resume.
#   RESUME=1 → force resume    FRESH=1 → force start over
#
# Usage:  ./scripts/testing/run_thesis_profiles.sh [phase00|phase01|all|start-over-all]
#         No argument on a terminal → interactive menu. No argument in a
#         pipe/CI → defaults to `all`.
#
#   all            phase00 + phase01 + debug in ONE pass. phase01 therefore runs
#                  BEFORE the winner is injected and uses the placeholder extractor —
#                  good for a plumbing check, wrong for real results.
#   start-over-all phase00 → winner injection → phase01, run SEQUENTIALLY and both
#                  from scratch (each leg forces FRESH=1, discarding saved progress).
#                  Stops before phase01 if phase00 did not finish cleanly. This is the
#                  option you want for a real end-to-end regeneration.
# Binary selection (any CMake profile):
#   auto: most recently built out/build/*/…/thesis
#   THESIS_BUILD=max-performance ./scripts/testing/run_thesis_profiles.sh phase00
#   THESIS_BIN=/abs/path/to/thesis ./scripts/testing/run_thesis_profiles.sh
set -u

cd "$(dirname "$0")/../.." # -> software/nn

# Locate the thesis binary under any CMake build profile.
# Auto-pick prefers the CPU (max-performance) build when it exists. This is the reference
# backend for the thesis — the same one the Guayaquil paper pipeline defaults to
# (01_guayaquil_run_article_profiles.sh), so both experiments report from one backend. It is also
# the right default on the merits: these profiles' networks are tiny (kernel-launch-bound on
# GPU), and "most recently built" used to silently switch runs to whatever was rebuilt last.
# Override with THESIS_BUILD/THESIS_BIN to target another backend.
if [ -n "${THESIS_BIN:-}" ]; then
    BIN="$THESIS_BIN"
elif [ -n "${THESIS_BUILD:-}" ]; then
    BIN="out/build/${THESIS_BUILD}/src/experiments/thesis/thesis"
elif [ -x "out/build/max-performance/src/experiments/thesis/thesis" ]; then
    BIN="out/build/max-performance/src/experiments/thesis/thesis"
else
    BIN=$(find out/build -maxdepth 5 -type f -name thesis \
              -path '*/experiments/thesis/thesis' -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
    echo "thesis binary not found. Build it, e.g.:"
    echo "  cmake --build out/build/max-performance --target thesis -j\$(nproc)"
    echo "or point at one:  THESIS_BIN=/path/to/thesis $0 $*"
    exit 1
fi
echo "using binary: $BIN"

# ── Parallelism (memory-gated) ───────────────────────────────────────────────
# Run several profiles at once when RAM allows. Each profile is an independent
# run (own result file by run_tag), so phase00/phase01 parallelise safely.
# Job count = min(free_RAM / per-job, nproc), capped at 4 to avoid GPU
# oversubscription. Override with THESIS_JOBS; tune per-job budget with
# THESIS_JOB_MEM_MB (default 5120 — measured snn-ae/poisson voice profiles peak
# ~4.4GB RSS+swap each, EEG profiles ~2.1GB; the old 2048 default let 4 voice
# jobs oversubscribe a 17GB box into heavy swap). JOBS=1 keeps the live
# progress-bar UX.
avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
[ -z "${avail_mb:-}" ] && avail_mb=2048
per_mb=${THESIS_JOB_MEM_MB:-5120}
jobs_mem=$(( avail_mb / per_mb )); [ "$jobs_mem" -lt 1 ] && jobs_mem=1
cpus=$(nproc)
if [ -n "${THESIS_JOBS:-}" ]; then
    JOBS=$THESIS_JOBS
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
  3) all     — phase00 + phase01 + debug, in ONE pass
               (phase01 uses the placeholder extractor — see 4)
  4) start over all — phase00 + phase01, from scratch, in order
               discards saved progress, runs phase00, injects the winner,
               THEN runs phase01 on it. This is the correct full pipeline.
  j) set parallel job count (currently $JOBS; auto-detected default $auto_jobs)
  b) (re)build the thesis binary first
  q) quit
MENU
            read -r -p "choice [1/2/3/4/j/b/q]: " ans
            case "$ans" in
                1) SCOPE=phase00; break ;;
                2) SCOPE=phase01; break ;;
                3) SCOPE=all;     break ;;
                4) SCOPE=start-over-all; break ;;
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
                    echo "building thesis …"
                    if cmake --build out/build/max-performance --target thesis -j"$(nproc)"; then
                        echo "build ok"
                    else
                        echo "build FAILED — fix errors, then choose again"
                    fi ;;
                q|Q) echo "aborted"; exit 0 ;;
                *)   echo "pick 1, 2, 3, 4, j, b, or q" ;;
            esac
        done
    else
        SCOPE=all   # non-interactive default
    fi
fi

# ── start-over-all: the full pipeline, from scratch, in dependency order ─────
# `all` runs phase00 and phase01 in ONE pass over a combined profile list, so phase01
# trains before the winner exists and silently uses the placeholder extractor. That is
# fine for a smoke/plumbing check and wrong for real results.
#
# This scope instead drives the two phases SEQUENTIALLY by re-invoking this same script:
#
#     phase00 (fresh) → post-processing injects the winner → phase01 (fresh)
#
# Re-invoking rather than restructuring the run loop means both passes get the identical
# monitor, checkpointing, failure capture and post-processing logic — there is no second
# code path to keep in sync.
#
# FRESH=1 on both legs is what makes this "start over": each phase clears its own
# .state file instead of resuming, so a previous partial run cannot leak in.
if [ "$SCOPE" = start-over-all ]; then
    SELF="scripts/testing/run_thesis_profiles.sh"  # cwd is software/nn (cd at the top of this file)

    # Carry the interactively-chosen job count into both legs; without this the children
    # re-derive it from free RAM and would silently ignore the menu's "j" setting.
    export THESIS_JOBS="$JOBS"

    echo
    echo "=== start over all: phase00 → (winner injection) → phase01 ==="
    echo "    both phases start from scratch (FRESH=1); saved progress is discarded."
    echo

    echo "--- leg 1/2: phase00 ---"
    if ! FRESH=1 "$SELF" phase00; then
        echo
        echo "!! phase00 did not finish cleanly — STOPPING before phase01."
        echo "!! Running phase01 now would train on the placeholder extractor rather than"
        echo "!! the Phase 00 winner, producing results that look valid but are not."
        echo "!! Fix the failures, then re-run (or use option 2 once phase00 is green)."
        exit 1
    fi

    echo
    echo "--- leg 2/2: phase01 (on the winner just injected) ---"
    FRESH=1 "$SELF" phase01
    rc=$?
    echo
    if [ "$rc" -eq 0 ]; then
        echo "=== start over all: COMPLETE — both phases finished cleanly ==="
        echo "    Review 'git diff src/experiments/thesis/profiles/phase01' (the winner injection)"
        echo "    and regenerate the thesis phase01 table:"
        echo "      python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py \\"
        echo "        --results-dir results/thesis/phase01 \\"
        echo "        --tables-dir  ../../documentation/00-thesis/monography/tables"
    else
        echo "=== start over all: phase00 succeeded, phase01 FAILED (exit $rc) ==="
    fi
    exit "$rc"
fi

case "$SCOPE" in
    phase00) ROOT="src/experiments/thesis/profiles/phase00" ;;
    phase01) ROOT="src/experiments/thesis/profiles/phase01" ;;
    all)     ROOT="src/experiments/thesis/profiles" ;;   # phase00 + phase01 + debug.json
    *) echo "usage: $0 [phase00|phase01|all|start-over-all]"; exit 1 ;;
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
mkdir -p results/thesis
STATE="results/thesis/run_profiles_${SCOPE}.state"
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
: > "$TMP/tally"        # one PASS/FAIL line per completed profile (parallel-safe)
: > "$TMP/completions"  # status/name/duration per completed profile (monitor "recent" panel)
mkdir -p "$TMP/active"
mkdir -p "$TMP/meta"    # cached per-profile metadata parsed from each log's head

# Stable, human-reachable per-profile logs (overwritten each run) so you can
# `tail -f results/thesis/run_logs/<profile>.log` to watch any single worker in full.
LOGDIR="results/thesis/run_logs"
mkdir -p "$LOGDIR"

# Live dashboard: with JOBS>1 the workers' own progress bars are hidden in their
# logs (concurrent bars can't share one terminal), so a monitor renders them in
# place. Each frame shows:
#
#   header   work-weighted % bar, done/total, pass/fail/skip, running count,
#            elapsed, ETA, projected finish CLOCK time, profiles/hour, and host
#            pressure (jobs/cpus, RAM, swap, load, which build is running)
#   workers  one 4-row panel per running profile: name + worker runtime; then
#            modality · strategy · model/loss · repeat i/n seed=s · dataset shape;
#            then the two live bars (outer epochs/folds, inner batches) with
#            pct, counts, loss and per-bar ETA
#   recent   the last 3 finished profiles with PASS/FAIL and wall time
#
# Tunables: THESIS_MONITOR=0 falls back to plain event lines; THESIS_MONITOR_INTERVAL
# sets the redraw period; THESIS_MONITOR_{TAIL,HEAD}_BYTES bound how much of each
# (multi-MB) worker log is re-read per frame.
MON=0
if [ "$JOBS" -gt 1 ] && [ "$tty_out" -eq 1 ] && [ "${THESIS_MONITOR:-1}" != 0 ]; then MON=1; fi

# ── Log scanning ────────────────────────────────────────────────────────────
# Worker logs grow to multiple MB (the binary rewrites its bars thousands of
# times). Reading a whole log per worker per frame would make the redraw cost
# scale with run length, so live state is parsed from a bounded TAIL and the
# static per-profile metadata is parsed once from the HEAD and cached.
MON_TAIL_BYTES="${THESIS_MONITOR_TAIL_BYTES:-65536}"
MON_HEAD_BYTES="${THESIS_MONITOR_HEAD_BYTES:-262144}"

# Strip CR-overwrites and ANSI colour so downstream matching sees plain text.
mon_clean() { tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g'; }

# Static metadata, parsed once per profile and cached: the binary prints
#   [Thesis] run_tag=<tag> modality=<m> strategy=<s> repeats=<n>
#   [Thesis] Loaded <n> samples from <n> subjects, <n> stimuli.
# near the start of the run. Emits KEY=VALUE lines. Only caches once the dataset
# line has appeared, so a worker scanned during startup is re-read next frame
# instead of caching a half-empty record forever.
mon_meta() {
    local name="$1" log="$2" out cache
    cache="$TMP/meta/$name"   # separate statement: on the same `local` line as name=,
                              # this would expand $name BEFORE the builtin assigns it
                              # (all args are expanded first) and resolve to "$TMP/meta/"
    if [ -s "$cache" ]; then cat "$cache"; return; fi
    out=$(head -c "$MON_HEAD_BYTES" "$log" 2>/dev/null | mon_clean | awk '
        /^\[Thesis\] run_tag=/ {
            for (i = 1; i <= NF; i++) {
                split($i, kv, "=")
                if (kv[1] == "modality") mod = kv[2]
                if (kv[1] == "strategy") strat = kv[2]
                if (kv[1] == "repeats")  reps = kv[2]
            }
        }
        /^\[Thesis\] Loaded/ {
            # "[Thesis] Loaded 1974 samples from 15 subjects, 11 stimuli."
            for (i = 1; i <= NF; i++) {
                if ($i == "samples")  smp  = $(i-1)
                if ($i ~ /^subjects/) subj = $(i-1)
                if ($i ~ /^stimuli/)  stim = $(i-1)
            }
        }
        END {
            printf "MOD=%s\nSTRAT=%s\nREPS=%s\nSMP=%s\nSUBJ=%s\nSTIM=%s\n",
                   mod, strat, reps, smp, subj, stim
        }')
    printf '%s\n' "$out"
    # Cache only a complete record (dataset line present ⇒ startup finished).
    printf '%s\n' "$out" | grep -q '^SMP=[0-9]' && printf '%s\n' "$out" > "$cache"
}

# Live state from the log tail, in ONE awk pass (one pipeline per worker per
# frame, not one per field). Emits:
#   REPEAT=<i>/<n> SEED=<s>          current repeat, from "=== Repeat i/n (seed=s) ==="
#   L<k>=<label> / D<k>=<data>       up to 2 most recent progress-bar pairs
# The binary renders each bar as a label line ("Autoencoder training │ ANN-AE
# (direct) │ MSE") followed by a data line ("<ascii bar> │ pct% n/m │ loss=… │
# ETA: …"). Pairing is done by looking back from each data line to the nearest
# preceding line that is not itself data/blank/log-chatter, which survives the
# metadata lines the binary interleaves between refreshes.
mon_scan() {
    tail -c "$MON_TAIL_BYTES" "$1" 2>/dev/null | mon_clean | awk '
        { line[NR] = $0 }
        $0 ~ /│/ && $0 ~ /[0-9]+(\.[0-9]+)?%/ { nd++; dl[nd] = NR }
        /^\[Thesis\] === Repeat/ {
            r = $0
            sub(/.*Repeat[ ]*/, "", r); sub(/[ ]*\(.*/, "", r); rep = r
            s = $0
            if (s ~ /seed=/) { sub(/.*seed=/, "", s); sub(/[^0-9].*/, "", s); seed = s }
        }
        function is_noise(t) {
            return (t ~ /^[[:space:]]*$/) || (t ~ /^\[Thesis\]/) || (t ~ /^Overall[ ]+\[/) \
                || (t ~ /│/ && t ~ /[0-9]+(\.[0-9]+)?%/) || (t ~ /^[0-9]{4}-[0-9]{2}-[0-9]{2} /)
        }
        END {
            if (rep != "") printf "REPEAT=%s\n", rep
            if (seed != "") printf "SEED=%s\n", seed
            k = 0
            for (i = (nd > 1 ? nd - 1 : 1); i <= nd; i++) {
                if (dl[i] == "") continue
                k++
                lbl = ""
                for (j = dl[i] - 1; j >= 1 && j >= dl[i] - 6; j--) {
                    if (!is_noise(line[j])) { lbl = line[j]; break }
                }
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
                printf "L%d=%s\nD%d=%s\n", k, lbl, k, line[dl[i]]
            }
        }'
}

# Split a bar LABEL ("Autoencoder training │ ANN-AE (direct) │ MSE") into its
# stage name and the model/loss annotations the binary appends after │.
mon_label_head() { local s="${1%%│*}"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }
mon_label_tail() {
    local s="$1"
    [[ "$s" != *"│"* ]] && { printf ''; return; }
    s="${s#*│}"; s="${s//│/ · }"
    s=$(printf '%s' "$s" | sed -E 's/[[:space:]]{2,}/ /g')
    s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"
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

# Background dashboard. Redraws in place (cursor-up) every THESIS_MONITOR_INTERVAL
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

# Wall-clock time a run is projected to end — for overnight runs this is the
# number you actually want ("done by 06:12"), which a bare "ETA 7:43:10" makes
# you compute yourself. Shows the weekday too once it lands past midnight.
fmt_finish() {
    local secs="$1" now_d tgt_d
    [ -z "$secs" ] && { printf '??:??'; return; }
    now_d=$(date +%j); tgt_d=$(date -d "+${secs} seconds" +%j 2>/dev/null) || { printf '??:??'; return; }
    if [ "$now_d" = "$tgt_d" ]; then date -d "+${secs} seconds" +%H:%M
    else date -d "+${secs} seconds" '+%a %H:%M'; fi
}

# Host pressure. These runs are memory-gated (see the JOBS heuristic above) and
# swap thrash is the usual reason a long run crawls, so the dashboard surfaces
# used/total RAM, swap-in-use and load average rather than hiding them in top.
sys_stats() {
    awk '
        /^MemTotal:/     { tot = $2 }
        /^MemAvailable:/ { avail = $2 }
        /^SwapTotal:/    { stot = $2 }
        /^SwapFree:/     { sfree = $2 }
        END {
            used = (tot - avail) / 1048576; totg = tot / 1048576
            swu = (stot - sfree) / 1048576
            printf "MEMUSED=%.1f\nMEMTOT=%.1f\nSWAPUSED=%.1f\n", used, totg, swu
        }' /proc/meminfo 2>/dev/null
    awk '{ printf "LOAD=%s\n", $1 }' /proc/loadavg 2>/dev/null
}

# Terminal geometry as "<rows> <cols>".
#
# Reads `stty size < /dev/tty` rather than `tput lines/cols`: the monitor runs in a
# BACKGROUND subshell, and there tput's size ioctl fails (its stdin is /dev/null) so it
# silently falls back to terminfo's static defaults — 80x24 — instead of the real size.
# That made every frame clamp to 80 columns no matter how wide the terminal actually was,
# and left the height clamp comparing against a fictitious 24 rows. stty reading /dev/tty
# directly reports the true size from both foreground and background.
term_size() {
    local sz
    sz=$(stty size < /dev/tty 2>/dev/null)
    if [ -n "$sz" ]; then printf '%s\n' "$sz"; return; fi
    printf '%s %s\n' "${LINES:-24}" "${COLUMNS:-80}"   # headless/CI fallback
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

# Wide progress bar for the run-level (not per-worker) line, drawn from a
# percentage. Separate from mon_mini_bar because the header has the whole
# terminal to work with and benefits from the extra resolution.
mon_wide_bar() {
    local pct="$1" width="${2:-24}" filled ii bar=""
    filled=$(printf '%.0f' "$pct" 2>/dev/null); : "${filled:=0}"
    [ "$filled" -lt 0 ] 2>/dev/null && filled=0
    [ "$filled" -gt 100 ] 2>/dev/null && filled=100
    filled=$((filled * width / 100))
    bar='\033[32m'
    for ((ii = 0; ii < width; ii++)); do
        [ "$ii" -eq "$filled" ] && bar+='\033[90m'
        if [ "$ii" -lt "$filled" ]; then bar+="#"; else bar+="-"; fi
    done
    bar+='\033[0m'
    printf '%b' "$bar"
}

monitor_loop() {
    local prev=0 interval="${THESIS_MONITOR_INTERVAL:-2}" id nm st now_e p f d n k ln cols rows
    local overall_eta eta_secs done_w pct thr log hidden sw dur
    local w_mod w_strat w_reps w_smp w_subj w_stim w_rep w_seed
    local w_l1 w_d1 w_l2 w_d2 kk vv sub head tail_ann lbl_var dat_var
    local memused memtot swapused load
    local -a lines rec
    printf '\033[?25l'   # hide cursor
    while [ -e "$TMP/mon.on" ]; do
        read -r rows cols < <(term_size)
        [ -z "$cols" ] && cols=80
        [ -z "$rows" ] && rows=24
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
        if [ "$done_w" -gt 0 ] && [ "$total_weight" -gt 0 ]; then
            eta_secs=$(( (now_e - start) * (total_weight - done_w) / done_w ))
            overall_eta="$(fmt_hms "$eta_secs")"
        else
            eta_secs=""; overall_eta="calculating"
        fi
        # Percent of WORK (not of profile count) — with a 20:1 cost ratio between
        # AE and handcrafted profiles, a count-based bar sits near zero for hours
        # and then races, which is exactly the "lame" progress this replaces.
        # The ternaries MUST stay parenthesised: in awk, an unparenthesised `>` in a
        # printf argument list parses as output redirection ("print to file 0"), not
        # comparison — awk then dies with a syntax error and the field renders empty.
        pct=$(awk -v dw="$done_w" -v tw="$total_weight" 'BEGIN{ printf "%.1f", (tw>0 ? dw*100.0/tw : 0) }')
        thr=$(awk -v dn="$d" -v el="$((now_e - start))" 'BEGIN{ printf "%.1f", (el>0 ? dn*3600.0/el : 0) }')

        eval "$(sys_stats)"
        memused="${MEMUSED:-?}"; memtot="${MEMTOT:-?}"; swapused="${SWAPUSED:-0}"; load="${LOAD:-?}"

        # ── header ───────────────────────────────────────────────────────────
        lines=("$(printf '\033[1;36m── e05 monitor ──\033[0m \033[1m%s\033[0m ── %s ─────' \
            "$SCOPE" "$(date +%T)")")
        lines+=("$(printf '  [%s] \033[1m%s%%\033[0m work · %d/%d profiles · \033[32m✓%d\033[0m \033[31m✗%d\033[0m \033[90m⊘%d\033[0m · %d running' \
            "$(mon_wide_bar "$pct" 24)" "$pct" "$d" "$npending" "$p" "$f" "$skip" \
            "$(ls "$TMP/active" 2>/dev/null | wc -l)")")
        lines+=("$(printf '  elapsed \033[1m%s\033[0m · ETA ~\033[35m%s\033[0m · finish ~\033[35m%s\033[0m · %s prof/h' \
            "$(fmt_hms $((now_e - start)))" "$overall_eta" "$(fmt_finish "$eta_secs")" "$thr")")
        lines+=("$(printf '  \033[90mjobs %d/%dcpu · mem %s/%sGB · swap %sGB · load %s · %s\033[0m' \
            "$JOBS" "$cpus" "$memused" "$memtot" "$swapused" "$load" "$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$BIN")")")")")")")

        # ── per-worker panels ────────────────────────────────────────────────
        lines+=("$(printf '\033[90m── workers ────────────────────────────────────────────\033[0m')")
        for id in $(ls "$TMP/active" 2>/dev/null | sort -n); do
            nm=$(sed -n '1p' "$TMP/active/$id" 2>/dev/null); [ -z "$nm" ] && continue
            st=$(sed -n '2p' "$TMP/active/$id" 2>/dev/null); [ -z "$st" ] && st=$now_e
            log="$LOGDIR/$nm.log"
            w_mod=""; w_strat=""; w_reps=""; w_smp=""; w_subj=""; w_stim=""
            w_rep=""; w_seed=""; w_l1=""; w_d1=""; w_l2=""; w_d2=""
            while IFS='=' read -r kk vv; do
                case "$kk" in
                    MOD) w_mod="$vv" ;; STRAT) w_strat="$vv" ;; REPS) w_reps="$vv" ;;
                    SMP) w_smp="$vv" ;; SUBJ) w_subj="$vv" ;; STIM) w_stim="$vv" ;;
                esac
            done < <(mon_meta "$nm" "$log")
            while IFS='=' read -r kk vv; do
                case "$kk" in
                    REPEAT) w_rep="$vv" ;; SEED) w_seed="$vv" ;;
                    L1) w_l1="$vv" ;; D1) w_d1="$vv" ;; L2) w_l2="$vv" ;; D2) w_d2="$vv" ;;
                esac
            done < <(mon_scan "$log")

            # line 1: profile name + how long this worker has been running
            lines+=("$(vis_trunc "$(printf '  \033[1;36m%s\033[0m  \033[35mrun %s\033[0m' \
                "$nm" "$(fmt_mmss $((now_e - st)))")" "$cols")")

            # line 2: what this profile IS (modality/strategy/dataset shape) and
            # where it is in the repeat loop — the context the old monitor
            # dropped entirely, so every worker row looked interchangeable.
            # The model/loss annotation ("ANN-AE (direct) · MSE") the binary appends
            # after │ on the outer label is per-profile, not per-bar, so it belongs on
            # this metadata line rather than on a row of its own under the bar — that
            # keeps every worker to a predictable 4 rows, which matters because the
            # frame has to fit the terminal height (see the clamp below).
            # Field order is by volatility, not by category: this line is clamped to the
            # terminal width, so whatever sits last is what gets cut on a narrow window.
            # The repeat counter changes during the run and is what you actually watch;
            # the dataset shape is static and safe to lose, so it goes last.
            tail_ann="$(mon_label_tail "$w_l1")"
            sub=""
            [ -n "$w_mod" ]   && sub="$w_mod"
            [ -n "$w_strat" ] && sub="${sub:+$sub · }$w_strat"
            [ -n "$tail_ann" ] && sub="${sub:+$sub · }$tail_ann"
            if [ -n "$w_rep" ]; then
                sub="${sub:+$sub · }\033[0;33mrepeat ${w_rep}\033[90m${w_seed:+ seed=$w_seed}"
            elif [ -n "$w_reps" ]; then
                sub="${sub:+$sub · }${w_reps} repeats"
            fi
            [ -n "$w_smp" ]   && sub="${sub:+$sub · }${w_smp} smp/${w_subj} subj/${w_stim} stim"
            [ -n "$sub" ] && lines+=("$(vis_trunc "$(printf '     \033[90m%b\033[0m' "$sub")" "$cols")")

            # lines 3-4: the two live bars (outer = epochs/folds, inner = batches),
            # each as "<stage name> [mini bar] pct n/m | loss | ETA".
            for kk in 1 2; do
                lbl_var="w_l$kk"; dat_var="w_d$kk"
                head="$(mon_label_head "${!lbl_var}")"
                [ -z "$head" ] && [ -z "${!dat_var}" ] && continue
                if [ -n "${!dat_var}" ]; then
                    lines+=("$(vis_trunc "$(printf '     \033[1m%-20.20s\033[0m %s' \
                        "$head" "$(mon_compact_bar "${!dat_var}")")" "$cols")")
                else
                    lines+=("$(vis_trunc "$(printf '     \033[1m%s\033[0m' "$head")" "$cols")")
                fi
            done
            [ -z "$w_d1" ] && [ -z "$w_d2" ] && \
                lines+=("$(vis_trunc "$(printf '     \033[90mstarting…\033[0m')" "$cols")")
        done

        # ── recently finished ────────────────────────────────────────────────
        # Only rendered if the terminal has spare rows: the redraw math below
        # assumes one physical row per logical line, so overflowing the screen
        # would break the in-place update (see the height clamp).
        if [ -s "$TMP/completions" ]; then
            mapfile -t rec < <(tail -3 "$TMP/completions" 2>/dev/null)
            if [ "$((${#lines[@]} + ${#rec[@]} + 1))" -lt "$((rows - 1))" ]; then
                lines+=("$(printf '\033[90m── recent ─────────────────────────────────────────────\033[0m')")
                for ln in "${rec[@]}"; do
                    IFS=$'\t' read -r sw nm dur <<< "$ln"
                    if [ "$sw" = PASS ]; then
                        lines+=("$(vis_trunc "$(printf '  \033[32m✓\033[0m %-34.34s \033[90m%s\033[0m' "$nm" "$dur")" "$cols")")
                    else
                        lines+=("$(vis_trunc "$(printf '  \033[31m✗\033[0m %-34.34s \033[90m%s\033[0m' "$nm" "$dur")" "$cols")")
                    fi
                done
            fi
        fi

        # Clamp the frame to the terminal height. The cursor-up redraw assumes
        # one physical row per logical line; a frame taller than the screen
        # scrolls, so the next redraw lands in the wrong place and the display
        # smears. Richer per-worker panels made this reachable (4 workers × 4
        # rows + headers), so it is enforced rather than assumed.
        if [ "${#lines[@]}" -gt "$((rows - 1))" ]; then
            hidden=$(( ${#lines[@]} - (rows - 2) ))   # count BEFORE truncating
            lines=("${lines[@]:0:$((rows - 2))}")
            lines+=("$(printf '\033[90m  … %d more line(s) hidden — enlarge the terminal\033[0m' "$hidden")")
        fi

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
    local slot="$1" f="$2" name log ec sw err t0 dur
    name=$(basename "$f" .json); log="$LOGDIR/$name.log"
    t0=$(date +%s)
    "$BIN" --config "$f" > "$log" 2>&1; ec=$?
    dur=$(fmt_mmss $(( $(date +%s) - t0 )))
    if [ "$ec" -eq 0 ]; then sw=PASS; else sw=FAIL; fi
    (
        flock 9
        printf '%s %s\n' "$sw" "$f" >> "$STATE"   # resume checkpoint
        echo "$sw" >> "$TMP/tally"
        # Feeds the monitor's "recent" panel: status, profile, wall time. Tab-separated
        # because profile names never contain tabs but the duration column should stay
        # trivially splittable.
        printf '%s\t%s\t%s\n' "$sw" "$name" "$dur" >> "$TMP/completions"
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
# Thesis mixes fast handcrafted extraction (trains nothing) with slow autoencoders / DSNN
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
        # Persistent top banner inside the profile's own TUI (see thesis / THESIS_OVERALL),
        # mirroring the Guayaquil runner: the per-process bars can't know the whole-run status,
        # so the runner computes it here and hands it to the binary the same way Guayaquil does.
        export THESIS_OVERALL="$(printf 'Overall  [%d/%d]  elapsed %s  ETA %s   (%s)' \
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
    unset THESIS_OVERALL
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
    # concurrent thesis processes, OOM-killing unrelated work).
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

# ── Phase 00 post-processing ────────────────────────────────────────────────
# Phase 00 only produces raw per-profile CSV/JSON; three steps turn that into the
# artefacts everything downstream consumes. They used to be copy-pasted by hand after
# every run, which is how the thesis tables ended up frozen on stale (pre-deletion)
# data — so they now run automatically here.
#
#   1. 01_thesis_phase00_rank.py   → winners.json. Selects on d_penalized, NOT raw
#      d_truth (a collapsed latent can game d_truth), and reports exact ties.
#   2. 02_thesis_apply_winner.py   → injects the real winner into the 32 phase01
#      profiles, replacing the daub4/lfcc placeholder they ship with.
#   3. thesis_build_phase00_paraconsistent_tables.py → regenerates the thesis tables.
#
# NOTE: step 2 REWRITES 32 git-tracked files under
# src/experiments/thesis/profiles/phase01/. Expect a git diff there after a phase00 run;
# that diff is the point (it is what phase01 then trains on), but review it before
# committing. THESIS_SKIP_POST=1 skips this whole block.
post_process_phase00() {
    local py rc=0
    # These three scripts are stdlib-only, so python3 suffices; still prefer the
    # project venv when present so every pipeline entry point uses one interpreter.
    py="python3"
    [ -x ".venv/bin/python3" ] && py=".venv/bin/python3"

    echo
    echo "=== phase00 post-processing ==="

    echo "[post] ranking phase00 → winners.json"
    "$py" scripts/pipeline/thesis/01_thesis_phase00_rank.py \
        --profiles-dir src/experiments/thesis/profiles/phase00 \
        --results-dir  results/thesis/phase00 \
        --out          results/thesis/phase00/winners.json || { rc=$?; return $rc; }

    echo "[post] injecting winner into the phase01 profiles"
    "$py" scripts/pipeline/thesis/02_thesis_apply_winner.py \
        --winners      results/thesis/phase00/winners.json \
        --profiles-dir src/experiments/thesis/profiles/phase01 || { rc=$?; return $rc; }

    echo "[post] regenerating the thesis phase00 tables"
    "$py" scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py \
        --results-dir results/thesis/phase00 \
        --tables-dir  ../../documentation/00-thesis/monography/tables || { rc=$?; return $rc; }

    echo "[post] done — review 'git diff src/experiments/thesis/profiles/phase01' before committing"
    return 0
}

# Gated on scope AND on a clean run: ranking over a partial result set can pick the
# wrong winner, and step 2 would then bake that wrong winner into 32 tracked profiles.
# Failing closed keeps that mistake out of the tree — fix the failures and re-run
# (resume skips everything that already passed). THESIS_FORCE_POST=1 overrides.
if [ -n "${THESIS_SKIP_POST:-}" ]; then
    [ "$SCOPE" = phase00 ] || [ "$SCOPE" = all ] && echo "(THESIS_SKIP_POST set — skipping phase00 post-processing)"
elif [ "$SCOPE" = phase00 ] || [ "$SCOPE" = all ]; then
    if [ "$fail" -gt 0 ] && [ -z "${THESIS_FORCE_POST:-}" ]; then
        echo
        echo "!! skipping phase00 post-processing: $fail profile(s) failed, so the ranking"
        echo "!! would be computed over an incomplete result set and could select the wrong"
        echo "!! winner — which step 2 would then write into 32 tracked phase01 profiles."
        echo "!! Fix the failures and re-run (completed profiles are skipped), or set"
        echo "!! THESIS_FORCE_POST=1 to run it anyway."
    else
        if [ "$SCOPE" = all ]; then
            echo
            echo "!! scope 'all' ran phase01 in the SAME pass as phase00, so phase01 used the"
            echo "!! placeholder extractor, not the winner injected below. Re-run phase01"
            echo "!! (./scripts/testing/run_thesis_profiles.sh phase01) for it to take effect."
        fi
        post_process_phase00 || echo "!! phase00 post-processing FAILED (exit $?) — artefacts may be stale"
    fi
fi

[ "$fail" -eq 0 ]
