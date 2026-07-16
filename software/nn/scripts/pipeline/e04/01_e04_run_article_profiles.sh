#!/usr/bin/env bash
# e04_run_article_profiles.sh — Experiment 04 full paper pipeline: build → run → aggregate.
#
# Builds the experiment04 binary (OpenCL preset), runs all four article
# profiles (LSTM-AE, SNN-dense, SNN-conv1d, SNN-recurrent), converts model
# artifacts to PyTorch .pt format, and aggregates results into paper CSVs.
#
# Output files:
#   results/article_{lstm_ae,snn_dense,snn_conv1d,snn_recurrent}_comparative_metrics.csv
#   <paper-data-dir>/article_*_*.dat  (pgfplots DAT files)
#
# Usage:
#   cd software/nn
#   ./scripts/pipeline/e04_run_article_profiles.sh
#
# Runtime: ~2.5 h total (LSTM ~10 min, each SNN ~45 min).
# Requires: cmake, OpenCL runtime, Python 3 with numpy.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BIN="out/build/max-performance-opencl/src/experiments/04/experiment04"
PROFILES=(
  "src/experiments/04/profiles/article-lstm-ae.json"
  "src/experiments/04/profiles/article-snn-dense.json"
  "src/experiments/04/profiles/article-snn-conv1d.json"
  "src/experiments/04/profiles/article-snn-recurrent.json"
)

echo "[article-run] configuring/building max-performance-opencl"
cmake --preset=max-performance-opencl
cmake --build --preset=max-performance-opencl -j"$(nproc)" --target experiment04

echo "[article-run] running article profiles"
for profile in "${PROFILES[@]}"; do
  echo "[article-run] profile: $profile"
  "$BIN" --comparative-config "$profile"
done

echo "[article-run] converting NPZ artifacts to PT"
python3 scripts/data/npz_to_pytorch.py --models-dir results/models || true

echo "[article-run] building paper aggregate CSV files"
python3 scripts/pipeline/e04_build_lstm_vs_snn_paper_data.py \
  --results-dir results \
  --data-dir /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/04/profiles

echo "[article-run] done"
