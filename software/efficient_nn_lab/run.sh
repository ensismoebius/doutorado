#!/usr/bin/env bash
# Launches Efficient Neural Networks Lab using this project's own venv,
# regardless of the caller's current working directory (avoids the
# "software/" shadowing-namespace-package trap: running python -m from one
# directory too high finds this folder itself instead of the installed
# package).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Ambiente virtual nao encontrado em .venv/. Criando..." >&2
    python3 -m venv --system-site-packages .venv
    ./.venv/bin/pip install -e . -q
fi

exec ./.venv/bin/python -m efficient_nn_lab "$@"
