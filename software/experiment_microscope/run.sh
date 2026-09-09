#!/usr/bin/env bash
# Launches the Experiment Microscope from this project's own venv, regardless
# of the caller's working directory.
#
# The `cd` first is not cosmetic: running `python -m experiment_microscope`
# from `software/` (one level up) makes Python treat the `software/` directory
# itself as a namespace package and the import resolves to the wrong thing.
# Anchoring to this script's directory avoids that trap (same reason as
# software/efficient_nn_lab/run.sh).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Virtual env not found at .venv/. Creating (system-site-packages: PySide6 etc. come from the OS python)..." >&2
    python3 -m venv --system-site-packages .venv
    ./.venv/bin/pip install -e . -q
fi

exec ./.venv/bin/python -m experiment_microscope "$@"
