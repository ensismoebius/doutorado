#!/usr/bin/env bash
set -eu -o pipefail

# Usage: ./scripts/run_ci_docker.sh
# Builds a Docker image and runs the CI build inside a container, mounting the
# current repository. Outputs are written to ./ci-output on the host.

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR="$REPO_ROOT/ci-output"
mkdir -p "$OUT_DIR"

IMAGE_NAME=nn-ci:local

echo "Building Docker image ${IMAGE_NAME}..."
docker build -f "$REPO_ROOT/Dockerfile.ci" -t "$IMAGE_NAME" "$REPO_ROOT"

echo "Running build in container... logs will be saved to ${OUT_DIR}"
docker run --rm -v "$REPO_ROOT":/work -w /work "$IMAGE_NAME" /bin/bash -lc "
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Debug -G Ninja || true; \
ninja -j\\$(nproc) 2>&1 | tee ../ci-ninja-full.log || true; \
echo 'Collecting ExternalProject logs...'; \
find . -type f \( -name '*-out.log' -o -name '*-err.log' -o -name 'nfft3-configure-*.log' \) -print -exec printf '\\n---- %s ----\\n' {} \\; -exec sed -n '1,800p' {} \\; || true; \
cp -a ../ci-ninja-full.log /work/ci-output/ || true; \
cp -a nfft3-prefix/src/nfft3-stamp/* /work/ci-output/ 2>/dev/null || true; \
cp -a _deps/*/src/*/stamp/* /work/ci-output/ 2>/dev/null || true; \
"

echo "Done. Logs are in ${OUT_DIR}"
