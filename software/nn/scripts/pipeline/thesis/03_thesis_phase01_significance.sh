#!/usr/bin/env bash
# 03_thesis_phase01_significance.sh — every significance test the thesis quotes, declared here.
#
# WHY A FIXED LIST. A test chosen after looking at the table is not a test: with 142
# configurations there is always a pair that differs "significantly". So the families of
# comparisons, and the reference of each, are written down here, before the results, and
# the thesis quotes these files and no others (thesis_build_text_numbers.py reads every
# results/thesis/phase01/significance_*.csv; a file this script did not write would show up
# there as an unknown key family).
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
# All on EER, flat CV, standardized features. Pairing: `corrected` (Nadeau-Bengio corrected
# repeated k-fold t-test on fold-matched values) wherever the two conditions are scored on
# the same held-out data; `independent` for seen vs unseen phrases, whose test sets differ
# (the script refuses to pair them). Holm is applied within each family.
#
# Usage (from software/nn):
#   scripts/pipeline/thesis/03_thesis_phase01_significance.sh \
#       src/experiments/thesis/profiles/phase01 results/thesis/phase01
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <phase01 profiles dir> <phase01 results dir>" >&2
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
  run "vsvoice_${text}" "e05_p01_dsnn_voice_${text}_flat_std" "p01_dsnn_[a-z]+_${text}_flat_std\$" corrected
  for cond in pron imag; do
    run "fusion_${cond}_${text}" "e05_p01_dsnn_eeg${cond}_${text}_flat_std" \
        "p01_dsnn_(eeg|early|late)${cond}_${text}_flat_std\$" corrected
  done
  for src in "${SOURCES[@]}"; do
    run "ctrl_${src}_${text}" "e05_p01_dsnn_${src}_${text}_flat_std" \
        "p01_(dsnn|ann|cosine)_${src}_${text}_flat_std\$" corrected
  done
  for kind in eeg early late; do
    run "cond_${kind}_${text}" "e05_p01_dsnn_${kind}pron_${text}_flat_std" \
        "p01_dsnn_${kind}(pron|imag)_${text}_flat_std\$" corrected
  done
  for sig in voice eegpron eegimag; do
    run "cmp_${sig}_${text}" "e05_p01_cmp-hc_${sig}_${text}" "p01_cmp-(hc|ae|pca)_${sig}_${text}\$" corrected
  done
done

for src in voice earlypron earlyimag latepron lateimag; do
  run "dysph_${src}" "e05_p01_dsnn_${src}_dep_flat_std" \
      "p01_(dsnn_${src}_dep_flat_std|dysph_${src}_(probes|test)_l100)\$" corrected
done

for src in "${SOURCES[@]}"; do
  run "text_${src}" "e05_p01_dsnn_${src}_dep_flat_std" "p01_dsnn_${src}_(dep|indep)_flat_std\$" independent
done
