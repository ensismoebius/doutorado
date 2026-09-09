import numpy as np
import pytest

from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.encoding_lab import EncodingLab

pytestmark = pytest.mark.skipif(not is_available(), reason="nn_microscope not built")


def test_encoding_lab_produces_spike_trains(qapp, first_fsdd_window):
    node, _a = first_fsdd_window()
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


def test_encoding_lab_cursor_survives_rerender(qapp, first_fsdd_window):
    from experiment_microscope.core.selection import SelectionState

    node, _a = first_fsdd_window()
    if node is None:
        pytest.skip("FSDD corpus not available")
    sel = SelectionState()
    lab = EncodingLab(DataRepository(), sel)
    lab.show_node(node, "meeting01")
    sel.set("timestep", 40, cascade=False)
    cursor = lab._cursor
    assert cursor is not None
    lab._seed.setValue(7)  # forces a full _render (layout cleared + rebuilt)
    assert lab._cursor is cursor  # same object, not leaked
    assert abs(cursor._line.value() - 40.0) < 1e-6
    assert cursor._line in lab._layout_widget.getItem(0, 0).items
