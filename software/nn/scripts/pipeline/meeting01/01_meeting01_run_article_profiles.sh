#!/usr/bin/env bash
# 01_guayaquil_run_article_profiles.sh — Experiment 04 full paper pipeline: build → run → aggregate.
#
# Builds the guayaquil binary, runs all four article profiles (LSTM-AE, SNN-dense,
# SNN-conv1d, SNN-recurrent), converts model artifacts to PyTorch .pt format, and aggregates
# results into paper CSVs.
#
# Output files:
#   results/article_{lstm_ae,snn_dense,snn_conv1d,snn_recurrent}_comparative_metrics.csv
#   <paper-data-dir>/article_*_*.dat  (pgfplots DAT files)
#
# Usage:
#   cd software/nn
#   ./scripts/pipeline/guayaquil/01_guayaquil_run_article_profiles.sh          # asks which build to use
#   GUAYAQUIL_BUILD=max-performance ./scripts/…/01_guayaquil_run_article_profiles.sh   # no prompt
#   SKIP_BUILD=1 GUAYAQUIL_BUILD=max-performance ./scripts/…            # reuse existing binary
#
# BUILD SELECTION — the backend is part of the measurement, not a convenience.
# The paper reports train_ms / infer_ms and latency, and
# 02_guayaquil_build_lstm_vs_snn_paper_data.py feeds those straight into its tables, so all four
# profiles must run on the SAME backend or the SNN-vs-LSTM timing comparison is
# apples-to-oranges. `max-performance` (CPU/XTensor) is the reference and the default, matching
# the thesis (run_thesis_profiles.sh also defaults to it). Choosing any other build prints a
# warning; the prompt exists so the choice is deliberate, not so that any build is equally
# valid for the paper's numbers.
#
#   GUAYAQUIL_BUILD=<preset>  choose non-interactively (skips the prompt; required in CI/pipes)
#   SKIP_BUILD=1        reuse the existing binary instead of configuring/building
#
# The first run of a preset also *configures* it (a few minutes); later runs are incremental.
#
# Runtime: ~2.5 h total (LSTM ~10 min, each SNN ~45 min).
# Requires: cmake, Python 3 with numpy. (An OpenCL runtime is needed only if you pick an
# OpenCL preset.)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Prefer the project venv (bootstrapped by cmake/VendorPythonEnv.cmake — has numpy/torch/
# etc.) over bare `python3`, whose identity depends on whatever's on the caller's PATH.
PY="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PY="$ROOT_DIR/.venv/bin/python3"
fi

# The backend all four profiles should share (and that the thesis also uses). Not a claim
# about what any historical run used — result summaries do not record the backend — but the
# single reference chosen going forward. Overriding it warns, so a mismatch is never silent.
REFERENCE_BUILD="max-performance"

# ── choose the build ─────────────────────────────────────────────────────────
# GUAYAQUIL_BUILD wins; otherwise ask on a terminal; otherwise (pipe/CI) use the reference.
if [[ -z "${GUAYAQUIL_BUILD:-}" ]]; then
  if [[ -t 0 ]]; then
    mapfile -t _presets < <(
      "$PY" - <<'PY'
import json
with open("CMakePresets.json") as f:
    print("\n".join(p["name"] for p in json.load(f).get("configurePresets", [])
                    if not p.get("hidden")))
PY
    )
    # Fall back rather than die confusingly under `set -u` if the preset list is unreadable.
    if (( ${#_presets[@]} == 0 )); then
      echo "[article-run] could not read CMakePresets.json — using ${REFERENCE_BUILD}" >&2
      GUAYAQUIL_BUILD="$REFERENCE_BUILD"
    else

    echo "Which build should run the article profiles?"
    echo
    echo "  The paper reports train_ms / infer_ms / latency, so this choice lands in its"
    echo "  tables — and all four profiles must share one backend to be comparable."
    echo "  '${REFERENCE_BUILD}' is the reference (also the thesis default); anything else"
    echo "  measures a different backend and must be reported as such."
    echo
    _default_idx=1
    for i in "${!_presets[@]}"; do
      _mark=""
      [[ "${_presets[$i]}" == "$REFERENCE_BUILD" ]] && { _mark="   <- reference (thesis default)"; _default_idx=$((i + 1)); }
      _built=""
      [[ -x "out/build/${_presets[$i]}/src/experiments/guayaquil/guayaquil" ]] && _built=" [built]"
      printf "  %2d) %s%s%s\n" "$((i + 1))" "${_presets[$i]}" "$_built" "$_mark"
    done
    echo
    read -rp "Choice [${_default_idx}]: " _choice
    _choice="${_choice:-$_default_idx}"
    if ! [[ "$_choice" =~ ^[0-9]+$ ]] || (( _choice < 1 || _choice > ${#_presets[@]} )); then
      echo "[article-run] invalid choice: $_choice" >&2
      exit 1
    fi
    GUAYAQUIL_BUILD="${_presets[$((_choice - 1))]}"
    fi
  else
    GUAYAQUIL_BUILD="$REFERENCE_BUILD"
    echo "[article-run] non-interactive — defaulting to the reference build ($GUAYAQUIL_BUILD)"
  fi
fi

BIN="out/build/${GUAYAQUIL_BUILD}/src/experiments/guayaquil/guayaquil"
echo "[article-run] build: ${GUAYAQUIL_BUILD}"

# Say it plainly rather than let a non-reference backend slip into the paper unnoticed.
if [[ "$GUAYAQUIL_BUILD" != "$REFERENCE_BUILD" ]]; then
  echo
  echo "[article-run] ⚠  '${GUAYAQUIL_BUILD}' is NOT the reference backend (${REFERENCE_BUILD})."
  echo "[article-run] ⚠  train_ms / infer_ms / latency feed the paper's tables directly, so"
  echo "[article-run] ⚠  these results describe THIS backend. Do not present them as the"
  echo "[article-run] ⚠  reference setup without saying so."
  echo
fi

PROFILES=(
  "src/experiments/guayaquil/profiles/article-snn-dense.json"
  "src/experiments/guayaquil/profiles/article-snn-conv1d.json"
  "src/experiments/guayaquil/profiles/article-snn-recurrent.json"
  "src/experiments/guayaquil/profiles/article-lstm-ae.json"
)

# SKIP_BUILD=1 reuses the existing binary instead of configuring/building. Only do this when
# you know it is current — a stale binary silently produces results for code you are no
# longer running.
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  if [[ ! -x "$BIN" ]]; then
    echo "[article-run] SKIP_BUILD=1 but no binary at $BIN" >&2
    echo "[article-run] run once without SKIP_BUILD to configure/build '${GUAYAQUIL_BUILD}'." >&2
    exit 1
  fi
  echo "[article-run] SKIP_BUILD=1 — reusing existing $BIN"
else
  # First run of a preset also *configures* it (a few minutes); later runs are incremental.
  echo "[article-run] configuring/building ${GUAYAQUIL_BUILD}"
  cmake --preset="${GUAYAQUIL_BUILD}"
  cmake --build --preset="${GUAYAQUIL_BUILD}" -j"$(nproc)" --target guayaquil
fi

if [[ ! -x "$BIN" ]]; then
  echo "[article-run] no guayaquil binary at $BIN after build" >&2
  exit 1
fi

# Overall progress across the whole 4-profile run. Each profile is a separate process with
# its own full-screen TUI, so we compute the outer status here and hand the finished line to
# the binary via GUAYAQUIL_OVERALL; the comparative TUI renders it as a persistent top line above
# its own bars (see GuayaquilExperiment). The ETA is work-weighted + EMA-smoothed (see run_eta.sh)
# rather than a naive per-profile mean — essential here, where one fast LSTM precedes three
# slow SNNs, so counting profiles equally would lurch badly at the first boundary.
source scripts/lib/run_eta.sh

# Relative cost of each profile (the "work" the ETA weights by). LSTM does one run; each SNN
# profile sweeps architectures x thresholds x alphas, ~4-5x the wall-clock. 2 vs 9 encodes
# that ratio; the EMA corrects it from real timings after the first of each kind completes.
profile_weight() { case "$1" in *snn*) echo 9 ;; *) echo 2 ;; esac; }

echo "[article-run] running article profiles"
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
  export GUAYAQUIL_OVERALL="$(printf 'Overall  [%d/%d]  elapsed %s  ETA %s   (%s)' \
      "$_i" "$_total" "$(fmt_hms "$_elapsed")" "$_eta" "$(basename "$profile" .json)")"
  echo "[article-run] profile $_i/$_total: $profile"
  echo "[article-run] $GUAYAQUIL_OVERALL"

  _p_start=$(date +%s)
  "$BIN" --comparative-config "$profile"
  eta_update "$(profile_weight "$profile")" "$(( $(date +%s) - _p_start ))"
  _done_w=$((_done_w + $(profile_weight "$profile")))
done
unset GUAYAQUIL_OVERALL
printf '[article-run] all %d profiles done in %s\n' "$_total" "$(fmt_hms $(( $(date +%s) - _start )))"

echo "[article-run] converting NPZ artifacts to PT"
"$PY" scripts/data/npz_to_pytorch.py --models-dir results/guayaquil/models || true

echo "[article-run] building paper aggregate CSV files"
"$PY" scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py \
  --results-dir results/guayaquil \
  --data-dir /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/guayaquil/profiles

echo "[article-run] done"
