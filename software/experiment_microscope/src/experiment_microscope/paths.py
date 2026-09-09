"""Filesystem anchors.

Everything the app reads lives under the repository's ``software/nn`` tree.
This module resolves those locations once so no other module hard-codes a
relative path.
"""

from __future__ import annotations

import os
from pathlib import Path

#   software/experiment_microscope/src/experiment_microscope/paths.py
#   -> parents[3] == software/
_SOFTWARE_DIR = Path(__file__).resolve().parents[3]

NN_DIR = _SOFTWARE_DIR / "nn"
RESULTS_DIR = NN_DIR / "results"
MEETING01_RESULTS = RESULTS_DIR / "meeting01"
THESIS_RESULTS = RESULTS_DIR / "thesis"
PARACONSISTENT_GA_RESULTS = RESULTS_DIR / "paraconsistentGA"

MEETING01_PROFILES = NN_DIR / "src" / "experiments" / "meeting01" / "profiles"
MEETING01_MONITOR_DIR = NN_DIR / "scripts" / "pipeline" / "meeting01"

#: Candidate build directories for the compiled ``nn_microscope`` module,
#: most-specific first.
NN_BINDING_SEARCH_DIRS = (
    NN_DIR / "out" / "build" / "python-bindings" / "src" / "bindings",
    NN_DIR / "out" / "build" / "max-performance" / "src" / "bindings",
)

#: Default thesis dataset source (``~`` expansion mirrors the C++
#: ``nn::utility::expand_home`` in ThesisDataset.cpp).
THESIS_DEFAULT_DB = Path(os.path.expanduser("~")) / "database.sqlite"

NN_BINDING_BUILD_HINT = (
    "cd software/nn && cmake --preset=python-bindings && "
    "cmake --build out/build/python-bindings --target nn_microscope"
)
