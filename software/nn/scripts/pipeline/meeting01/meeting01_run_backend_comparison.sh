#!/usr/bin/env bash
# meeting01_run_backend_comparison.sh — Experiment 04 XTensor vs OpenCL backend benchmark.
#
# Builds meeting01 with both the max-performance (XTensor CPU) and
# max-performance-opencl (GPU) presets, runs the article-backend-bench
# profile on each, and saves results as separate CSVs for comparison.
#
# Output files:
#   results/meeting01/article_backend_bench_xtensor_comparative_metrics.csv
#   results/meeting01/article_backend_bench_opencl_comparative_metrics.csv
#
# Usage:
#   cd software/nn
#   ./scripts/pipeline/meeting01/meeting01_run_backend_comparison.sh
#
# Requires: cmake, OpenCL runtime, XTensor dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PY="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PY="$ROOT_DIR/.venv/bin/python3"
fi

PROFILE="src/experiments/meeting01/profiles/article-backend-bench.json"
XT_BIN="out/build/max-performance/src/experiments/meeting01/meeting01"
OC_BIN="out/build/max-performance-opencl/src/experiments/meeting01/meeting01"
BASE_OUT="results/meeting01/article_backend_bench_comparative_metrics.csv"

echo "[backend-run] building xtensor preset"
cmake --preset=max-performance
cmake --build --preset=max-performance -j"$(nproc)" --target meeting01

echo "[backend-run] running xtensor benchmark"
"$XT_BIN" --comparative-config "$PROFILE"
cp "$BASE_OUT" "results/meeting01/article_backend_bench_xtensor_comparative_metrics.csv"

echo "[backend-run] building opencl preset"
cmake --preset=max-performance-opencl
cmake --build --preset=max-performance-opencl -j"$(nproc)" --target meeting01

echo "[backend-run] running opencl benchmark"
"$OC_BIN" --comparative-config "$PROFILE"
cp "$BASE_OUT" "results/meeting01/article_backend_bench_opencl_comparative_metrics.csv"

echo "[backend-run] updating backend table"
"$PY" scripts/pipeline/meeting01/02_meeting01_build_lstm_vs_snn_paper_data.py \
  --results-dir results \
  --data-dir /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/meeting01/data \
  --profiles-dir src/experiments/meeting01/profiles

echo "[backend-run] done"
