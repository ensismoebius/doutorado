"""End-to-end: pick a real FSDD window and follow it into the wavelet lab.

Requires the nn_microscope binding AND the FSDD corpus at the profile's
dataset_root. Skips cleanly otherwise.
"""

import numpy as np
import pytest

from experiment_microscope.data.meeting01_adapter import Meeting01Adapter
from experiment_microscope.processing._binding import is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="nn_microscope not built")


def test_meeting01_window_signal_is_zscored(first_fsdd_window):
    node, a = first_fsdd_window()
    if node is None:
        pytest.skip("FSDD corpus not available on this machine")
    sig = a.load_signal(node)
    assert sig.origin.value == "computed"
    x = np.asarray(sig.samples)
    assert x.shape == (256,)
    assert abs(x.mean()) < 1e-5 and abs(x.std() - 1.0) < 1e-3


def test_thesis_eeg_sample_signal_and_wavelet():
    from experiment_microscope.data.thesis_adapter import ThesisAdapter
    from experiment_microscope.paths import THESIS_DEFAULT_DB

    if not THESIS_DEFAULT_DB.is_file():
        pytest.skip("~/database.sqlite absent")
    a = ThesisAdapter()
    phases = a.children(a.root_nodes()[0])
    if not phases:
        pytest.skip("no thesis results")
    run = a.children(phases[0])[0]
    groups = [n for n in a.children(run) if n.handle.get("level") == "samples"]
    if not groups:
        pytest.skip("run has no live-sample group")
    sig = a.load_signal(a.children(groups[0])[2])
    assert sig.samples.ndim == 2 and sig.samples.shape[0] == 6
    from experiment_microscope.processing import wavelet as wl

    assert wl.decompose(sig.samples[0], "haar", "packet", 4).leaf_count == 16


@pytest.mark.slow
def test_thesis_run_feature_matrix_recompute():
    from experiment_microscope.data.thesis_adapter import ThesisAdapter
    from experiment_microscope.paths import THESIS_DEFAULT_DB

    if not THESIS_DEFAULT_DB.is_file():
        pytest.skip("~/database.sqlite absent")
    a = ThesisAdapter()
    phases = a.children(a.root_nodes()[0])
    if not phases:
        pytest.skip("no thesis results")
    runs = a.children(phases[0])
    hc = [r for r in runs if "hc_" in r.handle.get("run_tag", "")
          and (a._summary(r.handle["phase"], r.handle["run_tag"]).get("strategy") == "handcrafted")]
    if not hc:
        pytest.skip("no handcrafted phase00 run")
    fm = a.load_features(hc[0])
    assert fm.values.ndim == 2 and fm.values.shape[0] == len(fm.sample_labels)
    assert fm.origin.value == "computed"
    assert a.load_features(hc[0]) is fm  # cached


def test_window_feeds_wavelet_decompose(first_fsdd_window):
    node, a = first_fsdd_window()
    if node is None:
        pytest.skip("FSDD corpus not available")
    from experiment_microscope.processing import wavelet as wl

    sig = a.load_signal(node)
    d = wl.decompose(sig.samples, "haar", "packet", 4)
    assert d.leaf_count == 16
    assert d.subband_energies.size == 16
