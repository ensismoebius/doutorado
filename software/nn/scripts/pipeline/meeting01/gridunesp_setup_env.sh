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
# hdf5 is REQUIRED, not defensive: vendored matio (cmake/VendorMatio.cmake) needs
# HDF5 for MAT73 support and has no fallback -- `find_package(HDF5)` failing turns
# into a hard `CMake Error at .../matio-src/cmake/thirdParties.cmake: MAT73
# requires HDF5` during configure. Caught by scripts/pipeline/meeting01/
# run_gridunesp_docker_sim.sh (2026-09-17) -- omitting it here was silent until
# the very first GridUnesp configure attempt.
#
# fftw is not REQUIRED but avoids an unnecessary from-source vendored build:
# without it, cmake/VendorFFTW.cmake falls back to autotools-building FFTW3 from
# source (works, just slower and needs the autotools toolchain to behave under
# conda's compilers). Installing it from conda-forge is strictly cheaper.
#
# sqlite is REQUIRED for a reason specific to this environment: without a system
# SQLite3, cmake/VendorSqlite.cmake falls back to downloading a vendored
# amalgamation from a handful of hardcoded sqlite.org/GitHub URLs with no year
# component (e.g. sqlite.org/sqlite-amalgamation-3510300.zip). sqlite.org moves a
# version's amalgamation into a dated subdirectory once a newer release
# supersedes it, so those URLs 404 the moment upstream ships a new point release
# -- a latent, previously-unnoticed bug this validation surfaced (2026-09-17),
# independent of GridUnesp. Installing `sqlite` here makes `find_package(SQLite3
# QUIET)` succeed so that whole fallback path is never exercised.
#
# make is REQUIRED: neither `gcc_linux-64`/`gxx_linux-64` nor a minimal AlmaLinux
# base ship GNU make. Two things silently need it: (1) the vendored NFFT3 build
# (cmake/VendorNFFT3.cmake) runs its own `./configure && make`, and (2) GCC's own
# `-flto=auto` (the `max-performance` preset) spawns parallel LTRANS jobs via an
# internal `make -jN` -- with no `make` on PATH this fails deep inside
# collect2/lto-wrapper as `lto-wrapper: fatal error: execvp: No such file or
# directory`, a message that names neither "make" nor anything else recognisable
# as the missing piece. Caught by run_gridunesp_docker_sim.sh (2026-09-17).
#
# gxx_linux-64/gcc_linux-64 are pinned to 13, not 10: GCC 10 satisfies the
# `requires(...)` concepts floor (Tensor.hpp, Linear.hpp, Lif.hpp, Adam.hpp need
# real C++20 concepts, not the older Concepts TS), but that is NOT the project's
# actual minimum -- Meeting01Config.cpp uses `std::ostringstream::view()`, a C++20
# *library* feature (P2495) that GCC 10's libstdc++ does not implement yet, one
# compiler version short of concepts support. Building with `gxx_linux-64=10`
# fails deep in the dependency graph (91/138 build steps in, `meeting01_lib`) with
# `error: 'std::ostringstream' has no member named 'view'` -- easy to misdiagnose
# as a concepts problem since concepts-using code upstream of it compiles fine.
# Verified empirically (2026-09-17, run_gridunesp_docker_sim.sh) that GCC 13's
# libstdc++ has it; GCC 10's does not.
#
# sox is for ensure_datasets.sh: AudioMNIST ships at 48kHz and the profile expects
# the pre-resampled 8kHz corpus (audioMNIST_8k/); `sox in.wav -r 8000 -c 1 -b 16
# out.wav` is the resample step, run once per file during dataset setup, not during
# training itself.
#
# --override-channels: every package here comes from conda-forge, but conda
# still consults the default `channels:` list (pkgs/main, pkgs/r) during
# solving unless told not to. Recent conda refuses to run non-interactively
# if those default channels' Terms of Service have not been accepted
# (`CondaToSNonInteractiveError`) -- this breaks an unattended run even though
# nothing is ever actually installed from them. `--override-channels` makes
# conda-forge the only channel consulted, sidestepping the ToS gate entirely
# instead of accepting it.
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
  conda install -n "$ENV_NAME" -y --override-channels -c conda-forge \
    openblas pkg-config ninja git cmake ccache "gxx_linux-64=13" "gcc_linux-64=13" \
    zlib hdf5 fftw sqlite make sox
else
  echo "[gridunesp-setup] creating env '${ENV_NAME}'"
  conda create -n "$ENV_NAME" -y --override-channels -c conda-forge \
    openblas pkg-config ninja git cmake ccache "gxx_linux-64=13" "gcc_linux-64=13" \
    zlib hdf5 fftw sqlite make sox
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
