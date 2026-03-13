#!/usr/bin/env bash
set -eu -o pipefail

# Usage: ./scripts/collect_external_logs.sh [build-dir]
BUILD_DIR=${1:-build}
echo "Searching for ExternalProject logs in ${BUILD_DIR}..."
find "${BUILD_DIR}" -type f \( -name '*-out.log' -o -name '*-err.log' -o -name 'nfft3-configure-*.log' \) -print -exec printf '\n---- %s ----\n' {} \; -exec sed -n '1,800p' {} \;
