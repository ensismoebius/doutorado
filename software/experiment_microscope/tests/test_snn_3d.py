import numpy as np
import pytest

pytest.importorskip("PySide6")

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.snn_3d import Snn3D


def test_non_window_node_is_declined(qapp):
    v = Snn3D(DataRepository())
    v.show_node(TreeNode("x", "root", {"level": "root"}), "meeting01")
    assert "individual meeting01 window" in v._status.text()
    assert v._layers is None


def test_low_performance_mode_disables(qapp):
    class _AS:
        low_performance_mode = True

    v = Snn3D(DataRepository(), app_state=_AS())
    v.show_node(TreeNode("x", "w", {"level": "window"}), "meeting01")
    assert "low-performance" in v._status.text().lower()


def test_real_snn_3d_end_to_end(qapp, first_fsdd_window):
    from experiment_microscope.processing._binding import is_available

    if not is_available():
        pytest.skip("nn_microscope not built")
    node, adapter = first_fsdd_window()
    if node is None or not adapter._snn_model_specs():
        pytest.skip("no FSDD corpus or no trained *_encoder.npz")
    v = Snn3D(DataRepository())
    v.show_node(node, "meeting01")
    assert v._layers is not None
    assert len(v._layers) >= 3
    assert any(c["name"] == "LIF spikes" for c in v._layers)
    assert v._layers[0]["activity"].ndim == 1

    # unit spacing only — no cosmetic multiplier baked into node positions, so
    # the coordinate box drawn on screen reports the real layer/slot counts
    xmin, xmax, ymin, ymax, zmin, zmax = v._panel.plotter.bounds
    assert xmax == pytest.approx(len(v._layers) - 1, abs=1e-6)
    n_max = max(c["activity"].size for c in v._layers)
    assert ymax <= n_max / 2.0 + 1e-6


def test_timeline_flood_animation(qapp, first_fsdd_window):
    from experiment_microscope.core.animation import TimelinePlayer
    from experiment_microscope.processing._binding import is_available

    if not is_available():
        pytest.skip("nn_microscope not built")
    node, adapter = first_fsdd_window()
    if node is None or not adapter._snn_model_specs():
        pytest.skip("no FSDD corpus or trained model")
    player = TimelinePlayer()
    v = Snn3D(DataRepository())
    v.set_timeline(player)
    v.show_node(node, "meeting01")
    assert player.total_frames == len(v._layers)
    player.seek(1)
    assert v._active_upto == 1
    player.seek(player.total_frames - 1)
    assert v._active_upto == player.total_frames - 1
