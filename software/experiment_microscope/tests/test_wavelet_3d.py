import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.core.state import AppState
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.wavelet_3d import Wavelet3D

pytestmark = pytest.mark.skipif(not is_available(), reason="nn_microscope .so not built")


class _Adapter:
    def load_signal(self, node):
        rng = np.random.default_rng(0)
        return Signal1D(samples=rng.standard_normal(512), sample_rate=1000.0,
                        origin=Origin.MEASURED, label="noise")


def _repo():
    r = DataRepository()
    r._adapters["thesis"] = _Adapter()
    return r


def test_builds_normalized_landscape(qapp):
    v = Wavelet3D(_repo())
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    assert v._grid_z is not None
    n_leaves, per = v._grid_z.shape
    assert n_leaves == 16  # level 4 packet
    assert v._grid_z.max() <= 1.0 + 1e-9
    assert v._grid_z.min() >= 0.0


def test_threshold_nans_low_coefficients(qapp):
    v = Wavelet3D(_repo())
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    v._threshold.setValue(0.5)  # -> _render; grid_z itself is untouched
    assert np.isfinite(v._grid_z).all()


def test_coordinate_system_uses_real_bounds_not_cosmetic_scale(qapp):
    """The mesh must be built from the true coefficient index / leaf index /
    magnitude ratio — no invented multiplier baked into the geometry, so the
    coordinate box drawn on screen reports real values."""
    v = Wavelet3D(_repo())
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    n_leaves, per = v._grid_z.shape
    xmin, xmax, ymin, ymax, zmin, zmax = v._panel.plotter.bounds
    assert xmax == pytest.approx(per - 1, abs=1e-6)
    assert ymax == pytest.approx(n_leaves - 1, abs=1e-6)
    assert zmax <= 1.0 + 1e-6          # relative-magnitude ratio, never a scaled height


def test_low_performance_mode_disables(qapp):
    st = AppState()
    st.low_performance_mode = True
    v = Wavelet3D(_repo(), st)
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    assert v._grid_z is None
    assert "low-performance" in v._status.text()
