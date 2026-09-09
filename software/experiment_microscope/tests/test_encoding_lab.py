import numpy as np
import pytest

from experiment_microscope.data.meeting01_adapter import Meeting01Adapter
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.encoding_lab import EncodingLab

pytestmark = pytest.mark.skipif(not is_available(), reason="nn_microscope not built")


def _window_node(a):
    root = a.root_nodes()[0]
    fsdd = next(n for n in a.children(root) if n.label == "fsdd")
    fold0 = a.children(fsdd)[0]
    try:
        wg = next(n for n in a.children(fold0) if n.handle.get("level") == "windows")
        wins = a.children(wg)
    except RuntimeError:
        return None
    return wins[0] if wins else None


def test_encoding_lab_produces_spike_trains(qapp):
    a = Meeting01Adapter()
    node = _window_node(a)
    if node is None:
        pytest.skip("FSDD corpus not available")
    lab = EncodingLab(DataRepository())
    lab.show_node(node, "meeting01")
    assert lab._window is not None and lab._window.shape == (256,)
    poisson = lab._encode("poisson")
    latency = lab._encode("latency")
    assert set(np.unique(poisson)).issubset({0.0, 1.0})
    assert latency.shape == (256,) and np.all(np.isfinite(latency))
    # direct is the identity transform
    np.testing.assert_allclose(lab._encode("direct"), lab._window)


def test_encoding_lab_ignores_non_window(qapp):
    lab = EncodingLab(DataRepository())
    from experiment_microscope.data.adapters import TreeNode

    lab.show_node(TreeNode("run", "r", {"level": "run"}), "thesis")
    assert lab._window is None
