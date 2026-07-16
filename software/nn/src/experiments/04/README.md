# Experiment 04 — LSTM vs SNN Comparative Autoencoder

Profile-driven comparative experiment. Runs LSTM and SNN autoencoders side-by-side on
the same EEG/audio dataset, producing CSV metrics and pgfplots DAT files for the paper.

## What it does

1. Loads a JSON profile specifying model paradigm (`lstm` or `snn`), architecture, and training config
2. Runs k-fold cross-validation with the specified model
3. Writes per-fold metrics to `results/article_*_comparative_metrics.csv`
4. Optionally writes DAT files for LaTeX pgfplots

## Build

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target experiment04 -j$(nproc)
```

## Run

```bash
# Single profile
./out/build/max-performance/src/experiments/04/experiment04 \
  --comparative-config src/experiments/04/profiles/article-lstm-ae.json

# Full article pipeline (all 4 models, ~2.5 h)
./scripts/pipeline/e04/01_e04_run_article_profiles.sh
```

## Profiles (`profiles/`)

| Profile | Model | Est. runtime |
|---|---|---|
| `article-lstm-ae.json` | LSTM autoencoder | ~10 min |
| `article-snn-dense.json` | SNN, dense input transform | ~45 min |
| `article-snn-conv1d.json` | SNN, conv1d input transform | ~45 min |
| `article-snn-recurrent.json` | SNN, recurrent input transform | ~45 min |

`debug.json` and `debug_nested.json` are fast smoke-test profiles (few epochs/folds).

## Profile audit tests

```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

25 tests verify all 5 article profiles parse, validate, have `loss=mse`,
`seed_deterministic=false`, and consistent sweep arrays.

## Key source files

| File | Role |
|---|---|
| `experiment04.cpp` | Thin CLI entry point |
| `lib/include/ComparativeConfig.hpp` | Profile JSON parser |
| `lib/include/AutoencoderBuilders.hpp` | LSTM/SNN network builder |
| `lib/src/ComparativeDataset.cpp` | Dataset loading + z-score normalization |
| `lib/src/ComparativeEncoding.cpp` | Input transform (dense/conv1d/recurrent) |
| `lib/src/ComparativeTraining.cpp` | K-fold training loop |
| `lib/src/ComparativeOutput.cpp` | CSV and DAT writers |

## SNN architecture note

`snn_architectures: ["dense", "conv1d", "recurrent"]` in a profile selects the **input
transform**, not the network topology. All three share the same autoencoder network
(`linear:64:leaky → linear:32:identity` encoder, mirrored decoder). Only `linear`
layer specs are valid in `encoder_layer_spec` / `decoder_layer_spec`.

## Paper pipeline

```bash
# 1. Run all profiles
./scripts/pipeline/e04/01_e04_run_article_profiles.sh

# 2. Aggregate CSVs → paper DAT files (called automatically by step 1)
python3 scripts/pipeline/e04/02_e04_build_lstm_vs_snn_paper_data.py \
  --results-dir results \
  --data-dir /path/to/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/04/profiles

# 3. Compile paper
cd documentation/07-articlesProduced/conference71070Guaiaquil
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
