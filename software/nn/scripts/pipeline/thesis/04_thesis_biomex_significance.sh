#!/usr/bin/env bash
# 04_thesis_biomex_significance.sh — the significance tests of the BIOMEX-DB replication.
#
# WHY THESE AND NO OTHERS. The replication is confirmatory (.wiki/Experiments/Thesis.md sec. 11):
# it asks whether what the thesis concluded on the Coretto base holds on a second base. So its
# tests are not chosen here: they are 03_thesis_phase01_significance.sh's families, the same
# reference, the same selection, the same pairing, the same Holm correction, restricted to the
# sources BIOMEX-DB has (no imagined speech: no eegimag, no *imag fusion, so no `cond` family;
# no extractor comparison was run, so no `cmp` family). Written 2026-11-13, after the runs had
# finished and before any of their numbers was read.
#
#   family (file significance_<name>.csv)   reference        compared with it            Coretto question it re-asks
#   vsvoice_<text>                          voice            eegpron, earlypron, latepron does EEG or fusion beat voice?
#   fusion_pron_<text>                      pronounced EEG   its early and late fusion    does fusion beat EEG alone?
#   ctrl_<source>_<text>                    the DSNN         its ANN twin, cosine         does the network (the spiking) matter?
#   dysph_<source>                          the clean cell   full whisper, both scopes    does the degradation hurt?
#   text_<source>                           seen phrases     unseen phrases               Question 4
#
# Usage (from software/nn):
#   scripts/pipeline/thesis/04_thesis_biomex_significance.sh \
#       src/experiments/thesis/profiles/biomex results/thesis/biomex
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <biomex profiles dir> <biomex results dir>" >&2
  exit 2
fi
PROFILES="$1"
RESULTS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"

rm -f "$RESULTS"/significance_*.csv

run() { # name reference select pairing
  python3 "$HERE/thesis_cross_profile_significance.py" \
    --profiles-dir "$PROFILES" --results-dir "$RESULTS" --metric eer \
    --reference "$2" --select "$3" --pairing "$4" \
    --out "$RESULTS/significance_$1.csv" > /dev/null
  echo "[significance] $1"
}

SOURCES=(voice eegpron earlypron latepron)

for text in dep indep; do
  run "vsvoice_${text}" "e05_bx01_dsnn_voice_${text}_flat_std" "bx01_dsnn_[a-z]+_${text}_flat_std\$" corrected
  run "fusion_pron_${text}" "e05_bx01_dsnn_eegpron_${text}_flat_std" \
      "bx01_dsnn_(eeg|early|late)pron_${text}_flat_std\$" corrected
  for src in "${SOURCES[@]}"; do
    run "ctrl_${src}_${text}" "e05_bx01_dsnn_${src}_${text}_flat_std" \
        "bx01_(dsnn|ann|cosine)_${src}_${text}_flat_std\$" corrected
  done
done

for src in voice earlypron latepron; do
  run "dysph_${src}" "e05_bx01_dsnn_${src}_dep_flat_std" \
      "bx01_(dsnn_${src}_dep_flat_std|dysph_${src}_(probes|test)_l100)\$" corrected
done

for src in "${SOURCES[@]}"; do
  run "text_${src}" "e05_bx01_dsnn_${src}_dep_flat_std" "bx01_dsnn_${src}_(dep|indep)_flat_std\$" independent
done
