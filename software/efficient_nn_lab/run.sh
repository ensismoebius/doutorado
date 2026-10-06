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
fi

# Self-heal: the venv can exist (python present) while the package itself is
# missing or broken -- e.g. a half-finished `pip install -e .` that left an
# empty egg-info/dist-info behind with no actual RECORD, or a venv created
# without the install step ever completing. Checking only for
# .venv/bin/python (as this script used to) does not catch that: a real,
# confirmed failure was "No module named efficient_nn_lab" from a venv whose
# .venv/bin/python existed but had no pip of its own and no installed
# package. Actually importing the module is the only check that proves the
# install is genuinely usable.
if ! ./.venv/bin/python -c "import efficient_nn_lab" >/dev/null 2>&1; then
    echo "efficient_nn_lab nao esta instalado (ou instalacao quebrada) em .venv/. Reinstalando..." >&2
    # `python -m pip`, not `.venv/bin/pip`: this venv is --system-site-packages
    # and may not carry its own pip executable at all (observed 2026-10-06) --
    # `python -m pip` falls through to the system pip either way.
    ./.venv/bin/python -m pip install -e . -q
fi

exec ./.venv/bin/python -m efficient_nn_lab "$@"
