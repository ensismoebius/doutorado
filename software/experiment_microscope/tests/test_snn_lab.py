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
    assert "spikes" in txt and ("membrane" in txt or "membrane panel hidden" in txt)
    # 3 stacked plots (+1 membrane panel when a trained .npz exists for the fold)
    assert len(list(v._layout_widget.ci.items)) in (3, 4)


def test_snn_lab_reacts_to_vth(qapp, first_fsdd_window):
    node, _a = first_fsdd_window()
    if node is None:
        pytest.skip("FSDD corpus not available")
    v = SnnLab(DataRepository(), SelectionState())
    v.show_node(node, "meeting01")
    low = v._status.text()
    v._vth.setValue(2.5)  # far higher threshold -> fewer output spikes, forces re-render
    assert v._status.text() != low


def test_recurrent_lif_trace_matches_transform_and_recurrence():
    """v_mem is a real per-step trajectory; spikes are bit-identical to the
    plain recurrent transform (the refactor did not change the science)."""
    from experiment_microscope.processing import meeting01 as m

    rng = np.random.default_rng(1)
    enc = rng.standard_normal((256, 1)).astype(float)
    old = np.asarray(m.architecture_transform(enc, "recurrent", 0.9, 1.0)).reshape(-1)
    spk, v_mem = m.recurrent_lif_trace(enc, 0.9, 1.0)
    spk, v_mem = spk.reshape(-1), v_mem.reshape(-1)
    assert np.array_equal(old, spk)
    assert v_mem.shape == (256,)
    x = enc.reshape(-1)
    v1 = 0.9 * x[0] + x[1] - (1.0 if x[0] >= 1.0 else 0.0) * 1.0
    assert v_mem[1] == pytest.approx(v1, abs=1e-5)
    # a spike step's membrane is at or above threshold
    fired = np.flatnonzero(spk > 0.5)
    if fired.size:
        assert (v_mem[fired] >= 1.0 - 1e-4).all()


def test_snn_lab_ignores_non_window(qapp):
    v = SnnLab(DataRepository())
    v.show_node(TreeNode("run", "r", {"level": "run"}), "thesis")
    assert v._window is None
