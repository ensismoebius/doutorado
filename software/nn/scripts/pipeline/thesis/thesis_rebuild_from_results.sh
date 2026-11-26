#!/usr/bin/env bash
# thesis_rebuild_from_results.sh -- rebuild every file the thesis derives from the committed run
# outputs, in dependency order, and report whether the rebuild matches the repository.
#
# Why. The thesis numbers come from runs that cost months of CPU (results/thesis/, committed).
# Everything between those outputs and the PDF -- the regenerated profiles, the tables, the
# significance tests, the WORLD analyses, the base facts, the ~6500 numbers of the prose and the
# PDF itself -- is cheap and deterministic. This script redoes all of it, so a reader can check
# that the committed tables and text really follow from the committed results, without trusting
# anyone's word for it. With --check it fails when any rebuilt file differs from git.
#
#   committed inputs                       this script                  derived (tracked) files
#   results/thesis/<phase>/  ─┐
#   <databases>/*.sqlite     ─┼─► 1 calibration ... 9 latexmk ─► profiles/, tables/*.csv,
#   src/.../profiles/phase00 ─┘                                  thesis_numbers.tex, significance_*.csv,
#                                                                world_analysis/, monografia.pdf
#
# Usage (from anywhere; it works in software/nn):
#   scripts/pipeline/thesis/thesis_rebuild_from_results.sh                  # rebuild, then compare
#   scripts/pipeline/thesis/thesis_rebuild_from_results.sh --check          # ... exit 1 on a real difference
#   scripts/pipeline/thesis/thesis_rebuild_from_results.sh --no-calibration # skip step 1 (about 50 min)
#   scripts/pipeline/thesis/thesis_rebuild_from_results.sh --no-pdf         # stop before latexmk
#
# The calibration (step 1) is recomputed in a scratch folder and compared with the committed one;
# the later steps keep using the committed calibration.json. Its "seconds" field is how long the
# run of record took, and the thesis quotes it (cal.minutes), so a rebuild must not overwrite it.
#
# What "the same" means (thesis_compare_rebuild.py): a rebuilt file equals the committed one
# byte for byte, or, for a JSON file, value for value with numbers within a relative 1e-12. The
# second case is float summation order (another numpy or CPU moves the 16th digit), measured on
# world_floor.json on 2026-11-26; no number the thesis prints changes. Wall-clock fields are not
# results: "seconds" is ignored in the calibration comparison.
#
# Needs: ~/.venv/bin/python with numpy, scipy, praat-parselmouth and torch (CPU); python3 with
# numpy/scipy for the *_significance.sh scripts; latexmk + pdflatex + bibtex for the PDF; both
# bases in ~/Documentos/academico/UNESP/doutorado/databases (.wiki/Guides/Data-Bases.md). It
# trains nothing. Measured on the author's machine (2026-11-26, 12 cores): steps 2-11 about
# 5 min, the calibration 48 min on an idle machine (as in the run of record); result REPRODUCED.
#
# Nothing is skipped: a missing input stops the script at that step, with the step's name.
set -euo pipefail

CHECK=0
PDF=1
CAL=1
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --no-pdf) PDF=0 ;;
    --no-calibration) CAL=0 ;;
    *) echo "usage: $0 [--check] [--no-calibration] [--no-pdf]" >&2; exit 2 ;;
  esac
done

NN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$NN"
PY="$HOME/.venv/bin/python"
DB="$HOME/Documentos/academico/UNESP/doutorado/databases"
PROF=src/experiments/thesis/profiles
RES=results/thesis
TABLES=../../documentation/00-thesis/monography/tables
MONO=../../documentation/00-thesis/monography
T=scripts/pipeline/thesis
A=scripts/analysis

[[ -x "$PY" ]] || { echo "REFUSED: $PY is missing (the machine's shared Python env)" >&2; exit 1; }
for f in database.sqlite biomex.sqlite; do
  [[ -f "$DB/$f" ]] || { echo "REFUSED: $DB/$f is missing (.wiki/Guides/Data-Bases.md)" >&2; exit 1; }
done

STEP=0
step() { STEP=$((STEP + 1)); echo; echo "=== [$STEP] $* ($(date +%H:%M:%S))"; }
COMPARE=("$PY" $T/thesis_compare_rebuild.py)
DIFFERENT=0

if [[ $CAL -eq 1 ]]; then
  step "WORLD dysphonia calibration (Praat via parselmouth), in a scratch folder, vs the committed one"
  CALDIR="$(mktemp -d -t thesis-calibration.XXXXXX)"
  trap 'rm -rf "$CALDIR"' EXIT
  "$PY" $T/05_thesis_dysphonia_calibration.py --db "$DB/database.sqlite" \
    --subjects S10 S03 --jobs "$(nproc)" --out-dir "$CALDIR"
  echo -n "calibration.json: "
  "${COMPARE[@]}" --rel-tol 1e-12 --ignore-key seconds \
    $RES/dysphonia_calibration/calibration.json "$CALDIR/calibration.json" || DIFFERENT=1
  echo -n "measurements.csv: "
  "${COMPARE[@]}" --rel-tol 0 $RES/dysphonia_calibration/measurements.csv \
    "$CALDIR/measurements.csv" || DIFFERENT=1
else
  step "calibration NOT recomputed (--no-calibration): the committed calibration.json is used unchecked"
fi

step "Phase 00 ranking -> winners.json, winner_stability.csv"
"$PY" $T/01_thesis_phase00_rank.py --profiles-dir $PROF/phase00 --results-dir $RES/phase00 \
  --out $RES/phase00/winners.json --stability-out $RES/phase00/winner_stability.csv

step "every profile from the winners and the calibration -> $PROF/, profiles.manifest"
"$PY" $T/00_thesis_generate_profiles.py --winners $RES/phase00/winners.json --fused voice \
  --calibration $RES/dysphonia_calibration/calibration.json

step "tables of the fixed configuration (Phase 00, Phase 01, BIOMEX-DB)"
"$PY" $T/thesis_build_phase00_paraconsistent_tables.py \
  --profiles-dir $PROF/phase00 --results-dir $RES/phase00 --tables-dir $TABLES
"$PY" $T/thesis_build_phase01_auth_tables.py --protocol fixed \
  --profiles-dir $PROF/phase01 --results-dir $RES/phase01 --tables-dir $TABLES
"$PY" $T/thesis_build_biomex_tables.py --protocol fixed \
  --profiles-dir $PROF/biomex --results-dir $RES/biomex \
  --coretto-profiles-dir $PROF/phase01 --coretto-results-dir $RES/phase01 --tables-dir $TABLES

step "tables of the per-fold protocol (Coretto, BIOMEX-DB)"
"$PY" $T/thesis_build_phase01_auth_tables.py --protocol per-fold \
  --profiles-dir $PROF/perfold --results-dir $RES/perfold \
  --fixed-profiles-dir $PROF/phase01 --fixed-results-dir $RES/phase01 --tables-dir $TABLES
"$PY" $T/thesis_build_biomex_tables.py --protocol per-fold \
  --profiles-dir $PROF/pfbiomex --results-dir $RES/pfbiomex \
  --coretto-profiles-dir $PROF/perfold --coretto-results-dir $RES/perfold --tables-dir $TABLES

step "the declared significance families -> $RES/<phase>/significance_*.csv"
$T/03_thesis_phase01_significance.sh $PROF/phase01 $RES/phase01
$T/04_thesis_biomex_significance.sh $PROF/biomex $RES/biomex
$T/10_thesis_perfold_significance.sh $PROF/perfold $RES/perfold
$T/11_thesis_pfbiomex_significance.sh $PROF/pfbiomex $RES/pfbiomex

step "WORLD: identity pilot, identity margin, grid (both protocols), floor test"
"$PY" $T/06_thesis_identity_pilot.py --results $RES/pilot --out $RES/pilot/pilot_verdict.json
"$PY" $T/07_thesis_identity_margin.py \
  --control $RES/identity_margin/e05_pil_cosine_voice_control_probes \
  --whisper $RES/identity_margin/e05_p01_dysph_voice_probes_l25 \
  --out $RES/identity_margin/identity_margin_verdict.json
"$PY" $T/08_thesis_world_analysis.py --protocol fixed --features $RES/world_margin \
  --profiles-root $PROF --results-root $RES --out $RES/world_analysis
"$PY" $T/08_thesis_world_analysis.py --protocol per-fold --bases coretto2017,biomex2021 \
  --profiles-root $PROF --results-root $RES --out $RES/world_analysis
"$PY" $T/09_thesis_world_floor.py --features $RES/world_margin \
  --out $RES/world_analysis/world_floor.json

step "facts about the bases, the AE-input reference, the EEG band power"
"$PY" $A/thesis_base_facts.py --biomex "$DB/biomex.sqlite" --coretto "$DB/database.sqlite" \
  --out $RES/base_facts.csv
"$PY" $A/thesis_ae_input_reference.py --db "$DB/database.sqlite" --out $RES/ae_input_reference.csv
"$PY" $A/thesis_eeg_band_power.py --db "$DB/database.sqlite" --out $RES/eeg_band_power.csv

step "every number and checked claim of the prose -> $TABLES/thesis_numbers.tex"
"$PY" $T/thesis_build_text_numbers.py \
  --phase00-profiles $PROF/phase00 --phase00-results $RES/phase00 \
  --phase01-profiles $PROF/phase01 --phase01-results $RES/phase01 \
  --biomex-profiles $PROF/biomex --biomex-results $RES/biomex \
  --perfold-profiles $PROF/perfold --perfold-results $RES/perfold \
  --pfbiomex-profiles $PROF/pfbiomex --pfbiomex-results $RES/pfbiomex \
  --ae-input-reference $RES/ae_input_reference.csv --base-facts $RES/base_facts.csv \
  --dysphonia-calibration $RES/dysphonia_calibration/calibration.json \
  --identity-pilot $RES/pilot/pilot_verdict.json \
  --identity-margin $RES/identity_margin/identity_margin_verdict.json \
  --world-analysis $RES/world_analysis \
  --profiles-manifest $PROF/profiles.manifest \
  --eeg-band-power $RES/eeg_band_power.csv \
  --out $TABLES/thesis_numbers.tex

if [[ $PDF -eq 1 ]]; then
  step "the PDF -> $MONO/monografia.pdf"
  (cd "$MONO" && latexmk -pdf -interaction=nonstopmode -halt-on-error monografia.tex >/dev/null) ||
    { echo "REFUSED: latexmk failed; read $MONO/monografia.log" >&2; exit 1; }
fi

step "every rebuilt file against git"
# The PDF embeds its build date, so it always differs byte for byte; it is left out of the check.
TOP="$(git rev-parse --show-toplevel)"
NOISE=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  status="${line:0:2}" path="${line:3}"
  if [[ "$status" == " M" ]]; then
    committed="$(mktemp)"
    git -C "$TOP" show "HEAD:$path" > "$committed"
    verdict="$("${COMPARE[@]}" --rel-tol 1e-12 "$committed" "$TOP/$path")" && same=1 || same=0
    rm -f "$committed"
    echo "$path: $verdict"
    if [[ $same -eq 1 ]]; then NOISE+=("$TOP/$path"); else DIFFERENT=1; fi
  else
    echo "$path: git status '$status' (new, deleted or renamed)"
    DIFFERENT=1
  fi
done < <(git status --porcelain -- $PROF $RES $TABLES)

if [[ $DIFFERENT -eq 0 ]]; then
  echo "REPRODUCED: every rebuilt profile, result-derived file and table is the committed result"
  if [[ ${#NOISE[@]} -gt 0 ]]; then
    echo "(the files listed above differ only by float rounding; restore them with:"
    echo "  git checkout -- ${NOISE[*]} )"
  fi
else
  echo "NOT REPRODUCED: the files marked 'different' above are other results" >&2
  [[ $CHECK -eq 1 ]] && exit 1
fi
