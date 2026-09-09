"""Channel selector (§8) + feature-matrix cell readout (§29)."""

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.core.selection import SelectionState
from experiment_microscope.data.adapters import FeatureMatrix, Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.feature_matrix import FeatureMatrixView
from experiment_microscope.views.wavelet_lab import WaveletLab


class _FakeAdapter:
    def __init__(self, payload):
        self._payload = payload

    def load_signal(self, node):
        if isinstance(self._payload, Signal1D):
            return self._payload
        raise NotImplementedError

    def load_features(self, node):
        if isinstance(self._payload, FeatureMatrix):
            return self._payload
        raise NotImplementedError


def _repo_with(payload):
    repo = DataRepository()
    repo._adapters["thesis"] = _FakeAdapter(payload)
    return repo


def test_wavelet_channel_selector_populates_for_multichannel(qapp):
    sig = Signal1D(
        samples=np.random.default_rng(0).standard_normal((6, 512)),
        sample_rate=1024.0, origin=Origin.MEASURED,
        channel_names=tuple(f"ch{c}" for c in range(6)),
    )
    view = WaveletLab(_repo_with(sig), SelectionState())
    view.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    assert view._channel.count() == 6 and view._channel.isEnabled()
    view._channel.setCurrentIndex(2)  # must not raise
    assert view._tree.topLevelItemCount() == 16


def test_feature_matrix_cell_readout_sets_selection(qapp):
    values = np.arange(12, dtype=float).reshape(3, 4)
    fm = FeatureMatrix(
        values=values,
        feature_names=("a", "b", "c", "d"),
        sample_labels=("s0", "s1", "s2"),
        class_labels=(1, 1, 2),
        origin=Origin.COMPUTED,
        set_label="demo",
    )
    sel = SelectionState()
    view = FeatureMatrixView(_repo_with(fm), sel)
    view.show_node(TreeNode("run", "r", {"level": "run"}), "thesis")

    class _Evt:
        def scenePos(self):
            return view._img.mapToScene(2.5, 1.5)  # feature 2, sample 1

    view._on_click(_Evt())
    assert sel.get("feature") == 2
    assert "6" in view._status.text()  # values[1, 2] == 6


def test_signal_view_downsamples_long_signal(qapp):
    import numpy as np
    from experiment_microscope.core.state import AppState
    from experiment_microscope.core.integrity import Origin
    from experiment_microscope.data.adapters import Signal1D
    from experiment_microscope.data.repository import DataRepository
    from experiment_microscope.views.signal_view import SignalView

    st = AppState()
    v = SignalView(DataRepository(), None, st)
    v._render(Signal1D(samples=np.zeros(200_000), sample_rate=1.0, origin=Origin.MEASURED,
                       unit="x", label="long"))
    assert st.display_downsampled is True
    v._render(Signal1D(samples=np.zeros(500), sample_rate=1.0, origin=Origin.MEASURED,
                       unit="x", label="short"))
    assert st.display_downsampled is False
