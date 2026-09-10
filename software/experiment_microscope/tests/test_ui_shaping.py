"""Tabs shown only when applicable (§6) + hover readout / plot provenance (§29)."""

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.data.adapters import TreeNode


def _visible(w):
    return {w.tabs.tabText(i) for i in range(w.tabs.count()) if w.tabs.isTabVisible(i)}


def test_tabs_follow_selection(qapp):
    import experiment_microscope.app.workspace as W

    w = W.Workspace()
    w._update_tab_visibility({"level": "window"}, "meeting01")
    vis = _visible(w)
    assert {"SNN Lab", "Reconstruction", "Encoding Lab"} <= vis
    assert not ({"Feature Matrix", "Triangle", "NSGA-II"} & vis)
    # always-on tabs stay
    assert {"Comparison", "Ranking", "Pipeline"} <= vis

    w._update_tab_visibility({"level": "run"}, "thesis")
    vis = _visible(w)
    assert {"Feature Matrix", "Triangle"} <= vis
    assert not ({"SNN Lab", "Reconstruction", "Timeline"} & vis)


def test_hidden_current_tab_falls_back(qapp):
    import experiment_microscope.app.workspace as W

    w = W.Workspace()
    w._open_tab("SNN Lab")
    w._update_tab_visibility({"level": "run"}, "thesis")   # SNN Lab now hidden
    assert w.tabs.isTabVisible(w.tabs.currentIndex())


def test_hover_readout_reports_curve_values(qapp):
    import pyqtgraph as pg

    from experiment_microscope.views._plotinfo import HoverReadout, set_source

    plot = pg.PlotWidget()
    plot.plot(np.arange(10), np.arange(10) * 2.0, name="y")
    set_source(plot, "unit test")
    hr = HoverReadout(plot, x_label="sample")
    pi = plot.getPlotItem()

    class _Evt:
        def __init__(self, sp):
            self._sp = sp

        def __getitem__(self, _i):
            return self._sp

    scene_pt = pi.getViewBox().mapViewToScene(pg.QtCore.QPointF(4.0, 8.0))
    # make the plot think the cursor is inside it
    pi.sceneBoundingRect = lambda: pg.QtCore.QRectF(-1e6, -1e6, 2e6, 2e6)
    hr._on_move(_Evt(scene_pt))
    assert hr._text.isVisible()
    assert "y:" in hr._text.toPlainText() and "sample" in hr._text.toPlainText()
