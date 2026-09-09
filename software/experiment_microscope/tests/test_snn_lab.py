import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.snn_lab import SnnLab

pytestmark = pytest.mark.skipif(not is_available(), reason="nn_microscope not built")


def test_snn_lab_runs_lif_sweep(qapp, first_fsdd_window):
    node, _a = first_fsdd_window()
    if node is None:
        pytest.skip("FSDD corpus not available")
    v = SnnLab(DataRepository(), SelectionState())
    v.show_node(node, "meeting01")
    assert v._window is not None and v._window.shape == (256,)
    txt = v._status.text()
    assert "spikes" in txt and "v_mem not exposed" in txt
    # 3 stacked plots
    assert len(list(v._layout_widget.ci.items)) == 3


def test_snn_lab_reacts_to_vth(qapp, first_fsdd_window):
    node, _a = first_fsdd_window()
    if node is None:
        pytest.skip("FSDD corpus not available")
    v = SnnLab(DataRepository(), SelectionState())
    v.show_node(node, "meeting01")
    low = v._status.text()
    v._vth.setValue(2.5)  # far higher threshold -> fewer output spikes, forces re-render
    assert v._status.text() != low


def test_snn_lab_ignores_non_window(qapp):
    v = SnnLab(DataRepository())
    v.show_node(TreeNode("run", "r", {"level": "run"}), "thesis")
    assert v._window is None
