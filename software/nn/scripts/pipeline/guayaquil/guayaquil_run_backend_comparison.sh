#!/usr/bin/env bash
# guayaquil_run_backend_comparison.sh — Experiment 04 XTensor vs OpenCL backend benchmark.
#
# Builds guayaquil with both the max-performance (XTensor CPU) and
# max-performance-opencl (GPU) presets, runs the article-backend-bench
# profile on each, and saves results as separate CSVs for comparison.
#
# Output files:
#   results/guayaquil/article_backend_bench_xtensor_comparative_metrics.csv
#   results/guayaquil/article_backend_bench_opencl_comparative_metrics.csv
#
# Usage:
#   cd software/nn
#   ./scripts/pipeline/guayaquil/guayaquil_run_backend_comparison.sh
#
# Requires: cmake, OpenCL runtime, XTensor dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PY="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PY="$ROOT_DIR/.venv/bin/python3"
fi

PROFILE="src/experiments/guayaquil/profiles/article-backend-bench.json"
XT_BIN="out/build/max-performance/src/experiments/guayaquil/guayaquil"
OC_BIN="out/build/max-performance-opencl/src/experiments/guayaquil/guayaquil"
BASE_OUT="results/guayaquil/article_backend_bench_comparative_metrics.csv"

echo "[backend-run] building xtensor preset"
cmake --preset=max-performance
cmake --build --preset=max-performance -j"$(nproc)" --target guayaquil

echo "[backend-run] running xtensor benchmark"
"$XT_BIN" --comparative-config "$PROFILE"
cp "$BASE_OUT" "results/guayaquil/article_backend_bench_xtensor_comparative_metrics.csv"

echo "[backend-run] building opencl preset"
cmake --preset=max-performance-opencl
cmake --build --preset=max-performance-opencl -j"$(nproc)" --target guayaquil

echo "[backend-run] running opencl benchmark"
"$OC_BIN" --comparative-config "$PROFILE"
cp "$BASE_OUT" "results/guayaquil/article_backend_bench_opencl_comparative_metrics.csv"

echo "[backend-run] updating backend table"
"$PY" scripts/pipeline/guayaquil/02_guayaquil_build_lstm_vs_snn_paper_data.py \
  --results-dir results \
  --data-dir /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/guayaquil/profiles

echo "[backend-run] done"
