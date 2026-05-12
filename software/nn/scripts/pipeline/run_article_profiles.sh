#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
python3 scripts/pipeline/build_paper_data.py \
  --results-dir results \
  --data-dir /home/ensismoebius/Repos/doutorado/documentation/07-articlesProduced/conference71070Guaiaquil/data \
  --profiles-dir src/experiments/04/profiles

echo "[article-run] done"
