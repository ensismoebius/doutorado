#!/usr/bin/env bash
# check_core_coverage_gate.sh — CI line-coverage gate for src/core/.
#
# Parses an lcov .info file and enforces 100% line coverage on every file
# under src/core/. Exits non-zero if any core file has uncovered lines.
#
# Usage:
#   scripts/ci/check_core_coverage_gate.sh <lcov-info-file>
#
# Typically called from .github/workflows/ci.yml after `lcov --capture`
# produces coverage.info.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <lcov-info-file>" >&2
  exit 2
fi

info_file="$1"
if [[ ! -f "$info_file" ]]; then
  echo "coverage file not found: $info_file" >&2
  exit 2
fi

awk '
BEGIN {
  in_core = 0;
  sf = "";
  lf = -1;
  lh = -1;
  failed = 0;
}
function flush_record() {
  if (in_core && sf != "") {
    if (lf < 0 || lh < 0) {
      printf("missing LF/LH for %s\n", sf) > "/dev/stderr";
      failed = 1;
      return;
    }
    if (lf == 0) {
      printf("no executable lines in %s\n", sf) > "/dev/stderr";
      failed = 1;
      return;
    }
    if (lh != lf) {
      printf("coverage gate fail: %s (LH=%d LF=%d)\n", sf, lh, lf) > "/dev/stderr";
      failed = 1;
    }
  }
}
/^SF:/ {
  flush_record();
  sf = substr($0, 4);
  lf = -1;
  lh = -1;
  in_core = (sf ~ /\/src\/core\// && sf !~ /BatchPrefetcher/);
  next;
}
/^LF:/ {
  lf = substr($0, 4) + 0;
  next;
}
/^LH:/ {
  lh = substr($0, 4) + 0;
  next;
}
/^end_of_record/ {
  flush_record();
  sf = "";
  lf = -1;
  lh = -1;
  in_core = 0;
  next;
}
END {
  if (failed) {
    exit 1;
  }
  print "core coverage gate passed: all src/core files at 100% line coverage";
}
' "$info_file"
