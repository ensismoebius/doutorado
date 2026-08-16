#!/usr/bin/env bash
# Launched by the PDF's "run:" links (see presentation.md, phase 3) to open
# the companion software directly on a given demo. Takes one argument: a
# DemoModule.slug (e.g. "snn.lif" -- see software/efficient_nn_lab's
# core/demo.py and the per-demo `slug = "..."` class attributes).
#
# hyperref's "run:" resolves relative paths against the PDF's own
# directory, not the caller's cwd -- so this script resolves the software
# directory from its own location instead of assuming any particular cwd.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $(basename "$0") <demo-slug>" >&2
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../../software/efficient_nn_lab"

# Backgrounded so the PDF viewer's "run:" handler doesn't block waiting for
# the Qt app to exit.
exec ./run.sh --demo "$1" &
