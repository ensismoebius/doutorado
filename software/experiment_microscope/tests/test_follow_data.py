"""End-to-end: pick a real FSDD window and follow it into the wavelet lab.

Requires the nn_microscope binding AND the FSDD corpus at the profile's
dataset_root. Skips cleanly otherwise.
"""

import numpy as np
import pytest

from experiment_microscope.data.meeting01_adapter import Meeting01Adapter
from experiment_microscope.processing._binding import is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="nn_microscope not built")


def _first_window_node(adapter):
    root = adapter.root_nodes()[0]
    fsdd = next(n for n in adapter.children(root) if n.label == "fsdd")
    fold0 = adapter.children(fsdd)[0]
    try:
        windows_group = next(
            n for n in adapter.children(fold0) if n.handle.get("level") == "windows"
        )
        wins = adapter.children(windows_group)
    except RuntimeError:
        return None
    return wins[0] if wins else None


def test_meeting01_window_signal_is_zscored():
    a = Meeting01Adapter()
    node = _first_window_node(a)
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


def test_window_feeds_wavelet_decompose():
    a = Meeting01Adapter()
    node = _first_window_node(a)
    if node is None:
        pytest.skip("FSDD corpus not available")
    from experiment_microscope.processing import wavelet as wl

    sig = a.load_signal(node)
    d = wl.decompose(sig.samples, "haar", "packet", 4)
    assert d.leaf_count == 16
    assert d.subband_energies.size == 16
