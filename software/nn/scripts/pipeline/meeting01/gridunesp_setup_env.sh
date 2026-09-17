#!/usr/bin/env bash
# gridunesp_setup_env.sh — one-time toolchain bootstrap for the GridUnesp cluster
# (NCC/UNESP). Creates a conda env with everything the top-level CMake hard-requires
# that GridUnesp's module system does not provide (openblas via pkg-config, ninja,
# git) plus a C++20 compiler. See .wiki/Guides/GridUnesp-Deployment.md for why each
# package is here and what happens if this step is skipped.
#
# zlib is defensive: find_package(ZLIB REQUIRED) in
# src/core/data_loaders/CMakeLists.txt has no vendored fallback (unlike SQLite3,
# which tries the system package first and falls back to a vendored amalgamation).
# Most Linux base images already have it, but it costs nothing to guarantee.
#
# Usage (on the GridUnesp login node, once):
#   module load miniconda/24.4.0-libmamba   # module name confirmed 2026-09; re-check
#                                            # with `module avail miniconda` if it 404s
#   ./scripts/pipeline/meeting01/gridunesp_setup_env.sh
#
# Every later step (configure, build, and the sbatch job itself) does:
#   module load miniconda/24.4.0-libmamba && conda activate meeting01-build
set -euo pipefail

ENV_NAME="${ENV_NAME:-meeting01-build}"

if ! command -v conda >/dev/null 2>&1; then
  echo "gridunesp_setup_env.sh: no 'conda' in PATH -- run 'module load miniconda/24.4.0-libmamba' first" >&2
  exit 1
fi

if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "[gridunesp-setup] env '${ENV_NAME}' already exists -- updating packages"
  conda install -n "$ENV_NAME" -y -c conda-forge \
    openblas pkg-config ninja git cmake ccache "gxx_linux-64=10" "gcc_linux-64=10" \
    zlib
else
  echo "[gridunesp-setup] creating env '${ENV_NAME}'"
  conda create -n "$ENV_NAME" -y -c conda-forge \
    openblas pkg-config ninja git cmake ccache "gxx_linux-64=10" "gcc_linux-64=10" \
    zlib
fi

cat <<'EOF'

[gridunesp-setup] done. Before configuring/building, every shell needs:
  module load miniconda/24.4.0-libmamba
  conda activate meeting01-build

The env's own gcc/g++ (gxx_linux-64) are pinned ahead of the module gcc/10.2.0 on
PATH once activated -- if `pkg_check_modules(openblas)` still fails after
activation, run `pkg-config --list-all | grep -i blas` to confirm PKG_CONFIG_PATH
picked up the conda env (conda's activation script sets this automatically; only a
problem under a non-standard shell init).
EOF
