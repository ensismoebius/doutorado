"""Cursor / range sync across panels via SelectionState (FIXME §25)."""

import pytest

pg = pytest.importorskip("pyqtgraph")

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.views._timesync import TimeCursor


def test_line_move_writes_selection_and_syncs_peers(qapp):
    sel = SelectionState()
    w1, w2 = pg.PlotWidget(), pg.PlotWidget()
    c1 = TimeCursor(w1.getPlotItem(), sel)
    c2 = TimeCursor(w2.getPlotItem(), sel)

    c1._line.setPos(183.0)  # user drags the cursor in panel 1
    assert sel.get("timestep") == 183
    assert abs(c2._line.value() - 183.0) < 1e-6  # panel 2 followed


def test_selection_timestep_moves_cursor(qapp):
    sel = SelectionState()
    w = pg.PlotWidget()
    c = TimeCursor(w.getPlotItem(), sel)
    sel.set("timestep", 42, cascade=False)
    assert abs(c._line.value() - 42.0) < 1e-6


def test_region_drag_writes_time_range(qapp):
    sel = SelectionState()
    w = pg.PlotWidget()
    c = TimeCursor(w.getPlotItem(), sel)
    c._region.setRegion((120.0, 220.0))
    c._on_region_moved()
    assert sel.get("time_range") == (120, 220)


def test_reattach_keeps_cursor_after_clear(qapp):
    sel = SelectionState()
    w = pg.PlotWidget()
    plot = w.getPlotItem()
    c = TimeCursor(plot, sel)
    sel.set("timestep", 10, cascade=False)
    plot.clear()
    c.reattach()
    assert c._line in plot.items
    assert abs(c._line.value() - 10.0) < 1e-6
