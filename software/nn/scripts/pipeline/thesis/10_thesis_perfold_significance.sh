#!/usr/bin/env bash
# 10_thesis_perfold_significance.sh — the significance tests of the per-fold protocol (Coretto).
#
# WHY THESE AND NO OTHERS. The per-fold protocol (.wiki/Experiments/Thesis-Per-Fold-Protocol.md)
# pre-registered in its section 5, before any run: "the same comparisons, the same statistics as
# the current Phase 01". So no test is chosen here: these are 03_thesis_phase01_significance.sh's
# families, with the same reference, selection, pairing and Holm correction, read on the pf01_*
# profiles. Each per-fold cell mirrors one flat, standardized cell of the fixed grid, so every
# name loses its "_flat_std" and nothing else changes. Written 2026-11-19, after the perfold
# array (5331068) had finished: what this file adds is the renaming, not a choice.
#
#   family (file significance_<name>.csv)   reference              compared with it          question
#   vsvoice_<text>                          voice                  the 6 EEG/fusion sources  does EEG or fusion beat voice?
#   fusion_<cond>_<text>                    EEG of that condition  its early and late fusion does fusion beat EEG alone?
#   ctrl_<source>_<text>                    the DSNN               its ANN twin, cosine      does the network (the spiking) matter?
#   dysph_<source>                          the clean cell         full whisper, both scopes does the degradation hurt?
#   cmp_<signal>_<text>                     handcrafted            autoencoder, PCA          Question 3
#   text_<source>                           seen phrases           unseen phrases            Question 4
#   cond_<kind>_<text>                      pronounced EEG         imagined EEG              does the EEG condition matter? (kind eeg|early|late)
#
# Pairing as in 03: `corrected` (Nadeau-Bengio) where the two conditions are scored on the same
# held-out data, `independent` for seen vs unseen phrases. A per-fold profile's outer folds are
# the flat CV's (thesis_cross_profile_significance.py, outer_cv).
#
# Known identity, not a defect: in every fold of every pf01_dsnn_* cell the extractor chosen
# among all candidates was a handcrafted one, so pf01_cmp-hc_<signal>_<text> (same seed, same
# folds, the handcrafted candidates only) made the same choices and holds the same per-fold EERs
# as pf01_dsnn_<signal>_<text>. The cmp families compare the autoencoder and PCA families with it.
#
# Usage (from software/nn):
#   scripts/pipeline/thesis/10_thesis_perfold_significance.sh \
#       src/experiments/thesis/profiles/perfold results/thesis/perfold
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <perfold profiles dir> <perfold results dir>" >&2
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

SOURCES=(voice eegpron eegimag earlypron earlyimag latepron lateimag)

for text in dep indep; do
  run "vsvoice_${text}" "e05_pf01_dsnn_voice_${text}" "pf01_dsnn_[a-z]+_${text}\$" corrected
  for cond in pron imag; do
    run "fusion_${cond}_${text}" "e05_pf01_dsnn_eeg${cond}_${text}" \
        "pf01_dsnn_(eeg|early|late)${cond}_${text}\$" corrected
  done
  for src in "${SOURCES[@]}"; do
    run "ctrl_${src}_${text}" "e05_pf01_dsnn_${src}_${text}" \
        "pf01_(dsnn|ann|cosine)_${src}_${text}\$" corrected
  done
  for kind in eeg early late; do
    run "cond_${kind}_${text}" "e05_pf01_dsnn_${kind}pron_${text}" \
        "pf01_dsnn_${kind}(pron|imag)_${text}\$" corrected
  done
  for sig in voice eegpron eegimag; do
    run "cmp_${sig}_${text}" "e05_pf01_cmp-hc_${sig}_${text}" "pf01_cmp-(hc|ae|pca)_${sig}_${text}\$" corrected
  done
done

for src in voice earlypron earlyimag latepron lateimag; do
  run "dysph_${src}" "e05_pf01_dsnn_${src}_dep" \
      "pf01_(dsnn_${src}_dep|dysph_${src}_(probes|test)_l100)\$" corrected
done

for src in "${SOURCES[@]}"; do
  run "text_${src}" "e05_pf01_dsnn_${src}_dep" "pf01_dsnn_${src}_(dep|indep)\$" independent
done
