#!/usr/bin/env bash
# 01_paraconsistentGA_run_all_profiles.sh — full NSGA-II autoencoder-architecture search:
# build → run every population profile → summarise the Pareto fronts.
#
# Runs all 12 shipped profiles (ann/snn × eeg/voice/fused-{early,late} + snn loss variants),
# each one a separate population evolved independently (.wiki/Experiments/ParaconsistentGA-Design.md §5.3). Structure and the
# work-weighted overall-ETA banner are copied from the Guayaquil article runner.
#
# Output files (results/paraconsistentGA/):
#   pga_<run_tag>_individuals.csv   one row per distinct genome (depth, widths, α/β/G1/G2,
#                                   d_penalized mean+std, param_count, est_latency_ms, feasibility)
#   pga_<run_tag>_pareto.json       the final feasible Pareto front + run metadata
#
# Usage:
#   cd software/nn
#   ./scripts/pipeline/paraconsistentGA/01_paraconsistentGA_run_all_profiles.sh          # asks which build
#   PGA_BUILD=max-performance ./scripts/…/01_paraconsistentGA_run_all_profiles.sh          # no prompt
#   SKIP_BUILD=1 PGA_BUILD=max-performance ./scripts/…                                      # reuse binary
#   PGA_PROFILES_GLOB='*snn*eeg*' ./scripts/…                                               # subset by glob
#   CLEAN_RESULTS=1 ./scripts/…                                                             # wipe results dir first
#
# BUILD SELECTION — the backend is part of the measurement. d_penalized is backend-agnostic,
# but est_latency_ms / param_count / inference_cost are reported per run, so all profiles
# should share one backend to stay comparable. `max-performance` (CPU/XTensor) is the
# reference and default, matching the thesis and Guayaquil runners; any other build warns.
#
# Runtime: the shipped profiles now use population_size=32, generations=64 (≈10x the offspring
#   count of the earlier 16x12 setting on which the ~24 h/max_samples=550 figure was measured).
#   The eval cache trains each distinct expressed phenotype once, so realized cost is far below
#   the naive 10x, but the sweep WILL exceed 24 h at max_samples=550 — re-measure after the first
#   few generations, and lower dataset.max_samples / n_seeds / generations for faster iteration
#   (SNN ≈ 2.5x ANN; see .wiki/Experiments/ParaconsistentGA-Design.md §5.4.1 / §5.6).
#
# Requires: cmake, ninja. CPU preset — no GPU/OpenCL runtime needed.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PY="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then PY="$ROOT_DIR/.venv/bin/python3"; fi

REFERENCE_BUILD="max-performance"

# ── choose the build ─────────────────────────────────────────────────────────
# PGA_BUILD wins; otherwise ask on a terminal; otherwise (pipe/CI) use the reference.
if [[ -z "${PGA_BUILD:-}" ]]; then
  if [[ -t 0 ]]; then
    mapfile -t _presets < <(
      "$PY" - <<'PY'
import json
with open("CMakePresets.json") as f:
    print("\n".join(p["name"] for p in json.load(f).get("configurePresets", [])
                    if not p.get("hidden")))
PY
    )
    if (( ${#_presets[@]} == 0 )); then
      echo "[pga-run] could not read CMakePresets.json — using ${REFERENCE_BUILD}" >&2
      PGA_BUILD="$REFERENCE_BUILD"
    else
      echo "Which build should run the GA profiles?"
      echo
      echo "  d_penalized is backend-agnostic, but est_latency_ms / param_count are reported"
      echo "  per run, so keep one backend across profiles. '${REFERENCE_BUILD}' is the"
      echo "  reference (thesis/Guayaquil default); anything else measures a different backend."
      echo
      _default_idx=1
      for i in "${!_presets[@]}"; do
        _mark=""
        [[ "${_presets[$i]}" == "$REFERENCE_BUILD" ]] && { _mark="   <- reference"; _default_idx=$((i + 1)); }
        _built=""
        [[ -x "out/build/${_presets[$i]}/src/experiments/paraconsistentGA/paraconsistentGA" ]] && _built=" [built]"
        printf "  %2d) %s%s%s\n" "$((i + 1))" "${_presets[$i]}" "$_built" "$_mark"
      done
      echo
      read -rp "Choice [${_default_idx}]: " _choice
      _choice="${_choice:-$_default_idx}"
      if ! [[ "$_choice" =~ ^[0-9]+$ ]] || (( _choice < 1 || _choice > ${#_presets[@]} )); then
        echo "[pga-run] invalid choice: $_choice" >&2; exit 1
      fi
      PGA_BUILD="${_presets[$((_choice - 1))]}"
    fi
  else
    PGA_BUILD="$REFERENCE_BUILD"
    echo "[pga-run] non-interactive — defaulting to the reference build ($PGA_BUILD)"
  fi
fi

BIN="out/build/${PGA_BUILD}/src/experiments/paraconsistentGA/paraconsistentGA"
echo "[pga-run] build: ${PGA_BUILD}"

if [[ "$PGA_BUILD" != "$REFERENCE_BUILD" ]]; then
  echo
  echo "[pga-run] ⚠  '${PGA_BUILD}' is NOT the reference backend (${REFERENCE_BUILD})."
  echo "[pga-run] ⚠  est_latency_ms / param_count describe THIS backend; do not present"
  echo "[pga-run] ⚠  them as the reference setup without saying so."
  echo
fi

# ── build ────────────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  if [[ ! -x "$BIN" ]]; then
    echo "[pga-run] SKIP_BUILD=1 but no binary at $BIN" >&2
    echo "[pga-run] run once without SKIP_BUILD to configure/build '${PGA_BUILD}'." >&2
    exit 1
  fi
  echo "[pga-run] SKIP_BUILD=1 — reusing existing $BIN (ensure it is current)"
else
  echo "[pga-run] configuring/building ${PGA_BUILD}"
  cmake --preset="${PGA_BUILD}"
  cmake --build "out/build/${PGA_BUILD}" -j"$(nproc)" --target paraconsistentGA
fi

if [[ ! -x "$BIN" ]]; then
  echo "[pga-run] no paraconsistentGA binary at $BIN after build" >&2; exit 1
fi

# ── profiles ─────────────────────────────────────────────────────────────────
_glob="${PGA_PROFILES_GLOB:-*.json}"
mapfile -t PROFILES < <(find src/experiments/paraconsistentGA/profiles -maxdepth 1 -name "$_glob" | sort)
if (( ${#PROFILES[@]} == 0 )); then
  echo "[pga-run] no profiles matched '${_glob}' under src/experiments/paraconsistentGA/profiles" >&2
  exit 1
fi

# The GA now resumes from checkpoints (per-profile and per-generation), so a plain restart
# CONTINUES an interrupted sweep. CLEAN_RESULTS=1 is the deliberate override: wipe every result
# AND checkpoint so a fresh sweep never mixes with old numbers (the CLAUDE.md benchmark gotcha).
if [[ "${CLEAN_RESULTS:-0}" == "1" ]]; then
  echo "[pga-run] CLEAN_RESULTS=1 — removing results/paraconsistentGA"
  rm -rf results/paraconsistentGA
fi

# ── overall progress + ETA ────────────────────────────────────────────────────
# Work-weighted, EMA-smoothed ETA (run_eta.sh), not a per-profile mean: SNN profiles are
# ~2.5x the ANN wall-clock (measured, .wiki/Experiments/ParaconsistentGA-Design.md §5.4.1), so counting profiles equally would
# lurch at every ann/snn boundary. 5 vs 2 encodes that ratio; the EMA corrects it from real
# timings after the first of each kind finishes.
source scripts/lib/run_eta.sh
profile_weight() { case "$(basename "$1")" in pga_snn_*) echo 5 ;; *) echo 2 ;; esac; }

echo "[pga-run] running ${#PROFILES[@]} profile(s)"
_total=${#PROFILES[@]}
_start=$(date +%s)
_total_w=0; for _p in "${PROFILES[@]}"; do _total_w=$((_total_w + $(profile_weight "$_p"))); done
_done_w=0
_i=0
eta_reset
for profile in "${PROFILES[@]}"; do
  _i=$((_i + 1))
  _rem_w=$((_total_w - _done_w))
  _elapsed=$(( $(date +%s) - _start ))
  _rem_s=$(eta_remaining "$_rem_w")
  _eta=$([ -n "$_rem_s" ] && printf '~%s' "$(fmt_hms "$_rem_s")" || echo "calculating")
  export PGA_OVERALL="$(printf 'Overall  [%d/%d]  elapsed %s  ETA %s   (%s)' \
      "$_i" "$_total" "$(fmt_hms "$_elapsed")" "$_eta" "$(basename "$profile" .json)")"
  echo "[pga-run] profile $_i/$_total: $profile"
  echo "[pga-run] $PGA_OVERALL"

  # Profile-level resume: a profile whose final Pareto JSON already exists is complete
  # (its checkpoint artifacts were removed on success), so skip it. This makes a restarted
  # sweep continue where it stopped instead of redoing finished profiles. CLEAN_RESULTS=1
  # wiped everything above, so it never triggers there. In-progress profiles have no
  # pareto.json yet but keep a checkpoint the binary resumes from automatically.
  _run_tag="$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('run_tag') or d.get('experiment',{}).get('run_tag',''))" "$profile")"
  _pareto="results/paraconsistentGA/pga_${_run_tag}_pareto.json"
  if [[ -n "$_run_tag" && -f "$_pareto" ]]; then
    echo "[pga-run] ✓ already complete ($_pareto) — skipping"
    eta_update "$(profile_weight "$profile")" 0
    _done_w=$((_done_w + $(profile_weight "$profile")))
    continue
  fi

  _p_start=$(date +%s)
  "$BIN" --config "$profile"
  eta_update "$(profile_weight "$profile")" "$(( $(date +%s) - _p_start ))"
  _done_w=$((_done_w + $(profile_weight "$profile")))
done
unset PGA_OVERALL
printf '[pga-run] all %d profile(s) done in %s\n' "$_total" "$(fmt_hms $(( $(date +%s) - _start )))"

# ── summary: Pareto front of every population ─────────────────────────────────
echo "[pga-run] Pareto fronts:"
"$PY" - <<'PY'
import glob, json, os
rows = []
for p in sorted(glob.glob("results/paraconsistentGA/pga_*_pareto.json")):
    try:
        j = json.load(open(p))
    except Exception as e:
        print(f"  (unreadable: {p}: {e})"); continue
    pop = j.get("population", {})
    front = j.get("pareto_front", [])
    best = min(front, key=lambda x: x.get("d_penalized_mean", 9e9), default=None)
    bd = f"{best['d_penalized_mean']:.4f}" if best else "—"
    bw = "x".join(str(w) for w in best["genome"]["encoder_widths"]) if best else "—"
    rows.append((os.path.basename(p).replace("pga_", "").replace("_pareto.json", ""),
                 f"{pop.get('model','?')}/{pop.get('modality','?')}",
                 len(front), bd, bw))
if not rows:
    print("  (none found — did the run write to results/paraconsistentGA/?)")
else:
    print(f"  {'run':28s} {'pop':14s} {'front':>5s}  {'best d_pen':>10s}  best architecture")
    for r in rows:
        print(f"  {r[0]:28s} {r[1]:14s} {r[2]:5d}  {r[3]:>10s}  {r[4]}")
PY
echo "[pga-run] done — full results in results/paraconsistentGA/"
