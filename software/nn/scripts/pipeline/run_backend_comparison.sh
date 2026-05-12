#!/usr/bin/env bash
# run_backend_comparison.sh — XTensor vs OpenCL backend benchmark.
#
# Builds experiment04 with both the max-performance (XTensor CPU) and
# max-performance-opencl (GPU) presets, runs the article-backend-bench
# profile on each, and saves results as separate CSVs for comparison.
#
# Output files:
#   results/article_backend_bench_xtensor_comparative_metrics.csv
#   results/article_backend_bench_opencl_comparative_metrics.csv
#
# Usage:
#   cd software/nn
#   ./scripts/pipeline/run_backend_comparison.sh
#
# Requires: cmake, OpenCL runtime, XTensor dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROFILE="src/experiments/04/profiles/article-backend-bench.json"
XT_BIN="out/build/max-performance/src/experiments/04/experiment04"
OC_BIN="out/build/max-performance-opencl/src/experiments/04/experiment04"
BASE_OUT="results/article_backend_bench_comparative_metrics.csv"

echo "[backend-run] building xtensor preset"
cmake --preset=max-performance
cmake --build --preset=max-performance -j"$(nproc)" --target experiment04

echo "[backend-run] running xtensor benchmark"
"$XT_BIN" --comparative-config "$PROFILE"
cp "$BASE_OUT" "results/article_backend_bench_xtensor_comparative_metrics.csv"

echo "[backend-run] building opencl preset"
cmake --preset=max-performance-opencl
cmake --build --preset=max-performance-opencl -j"$(nproc)" --target experiment04

echo "[backend-run] running opencl benchmark"
"$OC_BIN" --comparative-config "$PROFILE"
cp "$BASE_OUT" "results/article_backend_bench_opencl_comparative_metrics.csv"

echo "[backend-run] updating backend table"
python3 scripts/build_paper_data.py \
  --results-dir results \
  --data-dir /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/04/profiles

echo "[backend-run] done"
