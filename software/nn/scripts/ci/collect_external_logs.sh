#!/usr/bin/env bash
# collect_external_logs.sh — Dump CMake ExternalProject build logs to stdout.
#
# Finds all *-out.log, *-err.log, and nfft3-configure-*.log files produced by
# CMake ExternalProject_Add steps and prints them (up to 800 lines each).
# Useful for diagnosing third-party build failures in CI.
#
# Usage:
#   scripts/ci/collect_external_logs.sh [build-dir]
#
# build-dir defaults to "build" if omitted.
set -eu -o pipefail

# Usage: ./scripts/collect_external_logs.sh [build-dir]
BUILD_DIR=${1:-build}
echo "Searching for ExternalProject logs in ${BUILD_DIR}..."
find "${BUILD_DIR}" -type f \( -name '*-out.log' -o -name '*-err.log' -o -name 'nfft3-configure-*.log' \) -print -exec printf '\n---- %s ----\n' {} \; -exec sed -n '1,800p' {} \;
