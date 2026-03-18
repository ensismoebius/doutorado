#!/usr/bin/env bash
set -euo pipefail
# Format staged C/C++ files and re-add them to the index.
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

# Gather staged files (Added/Copied/Modified)
staged_files=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$staged_files" ]; then
  exit 0
fi

# Filter to C/C++ sources and headers
to_format=$(printf "%s\n" "$staged_files" | grep -E '\\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$' || true)

if [ -n "$to_format" ]; then
  printf "%s\n" "$to_format" | xargs --no-run-if-empty clang-format -i
  printf "%s\n" "$to_format" | xargs --no-run-if-empty git add
fi

exit 0
