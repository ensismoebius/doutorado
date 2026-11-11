"""Scientific-correctness gate (FIXME §49).

The GUI's live recompute of the thesis Phase-00 chain must reproduce a
persisted result exactly — otherwise the "microscope" is showing different
numbers than the experiment did.

Slow (full EEG feature extraction, ~30 s) and needs ``~/database.sqlite`` +
the persisted phase00 CSV; skipped otherwise. Run explicitly with:

    pytest -q tests/test_thesis_parity.py
"""

import csv
import json

import pytest
from experiment_microscope.paths import NN_DIR, THESIS_DEFAULT_DB, THESIS_RESULTS
from experiment_microscope.processing._binding import is_available

_REF = THESIS_RESULTS / "phase00" / "e05_e05_p00_hc_daub10_lfcc_c1_eeg_rep0_paraconsistent.csv"
# The profile that produced _REF: the spec is read from it, never inferred from the tag.
_PROFILE = (NN_DIR / "src" / "experiments" / "thesis" / "profiles" / "phase00"
            / "p00_hc_daub10_lfcc_c1_eeg.json")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_available(), reason="nn_microscope not built"),
    pytest.mark.skipif(not THESIS_DEFAULT_DB.is_file(), reason="~/database.sqlite absent"),
    pytest.mark.skipif(not _REF.is_file(), reason="reference phase00 CSV absent"),
]


def test_live_recompute_matches_persisted_phase00_row():
    from experiment_microscope.processing.thesis import (
        HandcraftedSpec,
        rank_handcrafted,
    )

    with _REF.open(newline="") as fh:
        ref = next(csv.DictReader(fh))

    profile = json.loads(_PROFILE.read_text())
    spec = HandcraftedSpec.from_config(profile)
    results = dict(rank_handcrafted(spec, root=str(THESIS_DEFAULT_DB),
                                    max_samples=profile["dataset"]["max_samples"]))
    values = results[ref["label"]]

    for field in ("alpha", "beta", "g1", "g2", "d_truth", "d_penalized"):
        got = float(values[field].magnitude)
        want = float(ref[field])
        assert abs(got - want) < 1e-6, f"{field}: live {got} vs persisted {want}"
