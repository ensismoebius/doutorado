#!/usr/bin/env bash
set -eu -o pipefail

# run_gridunesp_docker_sim.sh — validate the GridUnesp toolchain recipe
# (.wiki/Guides/GridUnesp-Deployment.md §1-2) inside a local AlmaLinux
# container before spending real queue time on the cluster.
#
# Builds Dockerfile.gridunesp-sim, then inside the container:
#   1. rsyncs the repo into a container-local directory (read-only mount ->
#      writable copy, so the host's own out/ build tree is never touched).
#   2. runs scripts/pipeline/meeting01/gridunesp_setup_env.sh UNMODIFIED.
#   3. `cmake --preset=max-performance` (the login-node step).
#   4. `cmake --build ... --target meeting01` (the compute-node step).
#   5. `meeting01 --help` as a binary smoke check.
#
# Does not simulate Slurm/job-nanny or the exact cluster CPU -- see the
# Dockerfile's own header comment for the full list of what this does and
# does not reproduce.
#
# Usage: ./scripts/pipeline/meeting01/run_gridunesp_docker_sim.sh

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
IMAGE_NAME=nn-gridunesp-sim:local

echo "[gridunesp-sim] building image ${IMAGE_NAME}..."
docker build -f "$REPO_ROOT/Dockerfile.gridunesp-sim" -t "$IMAGE_NAME" "$REPO_ROOT"

echo "[gridunesp-sim] running toolchain validation in container..."
docker run --rm -v "$REPO_ROOT":/src:ro -w /work "$IMAGE_NAME" /bin/bash -lc '
set -eu -o pipefail

echo "[gridunesp-sim] copying repo into container-local workspace (keeps host out/ untouched)..."
rsync -a --exclude out --exclude results --exclude ci-output --exclude .git /src/ /work/nn/
cd /work/nn

echo "[gridunesp-sim] running gridunesp_setup_env.sh unmodified..."
./scripts/pipeline/meeting01/gridunesp_setup_env.sh

source /opt/miniconda3/etc/profile.d/conda.sh
# conda'\''s activate.d scripts (e.g. binutils_linux-64) reference variables they
# never bothered making `nounset`-safe -- relax -u for the activation call only.
set +u
conda activate meeting01-build
set -u

echo "[gridunesp-sim] configuring (simulates the login-node step)..."
cmake --preset=max-performance

echo "[gridunesp-sim] building the meeting01 target (simulates the compute-node step)..."
cmake --build out/build/max-performance --target meeting01 -j"$(nproc)"

echo "[gridunesp-sim] smoke-checking the binary..."
out/build/max-performance/src/experiments/meeting01/meeting01 --help

echo "[gridunesp-sim] PASSED -- toolchain + configure + build succeed on a GridUnesp-like AlmaLinux + conda environment."
'
