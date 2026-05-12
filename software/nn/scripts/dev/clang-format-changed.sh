#!/usr/bin/env bash
# clang-format-changed.sh — Format staged C/C++ files and re-stage them.
#
# Runs clang-format (via .clang-format at repo root) on every C/C++ file
# that is currently staged (Added/Copied/Modified) and re-adds each file
# to the git index.  Exits 0 with no output when nothing is staged.
#
# Usage:
#   Invoke directly or install as a pre-commit hook:
#     cp scripts/dev/clang-format-changed.sh .git/hooks/pre-commit
#   Or reference via .githooks/pre-commit-template.
#
# Requires: clang-format on PATH.
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
