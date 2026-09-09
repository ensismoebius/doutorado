"""Shared time cursor + range across every time-domain plot (FIXME §25).

A view that shows a signal against a sample/time axis creates one
``TimeCursor`` bound to its pyqtgraph ``PlotItem`` and the window's shared
``SelectionState``. Dragging the cursor line in any panel writes
``selection.timestep``; every other ``TimeCursor`` moves to match. Dragging the
shaded region writes ``selection.time_range``.

The cursor carries no meaning of its own — it is a pointer the panels agree on.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.views._pg import PG_OK, pg


class TimeCursor:
    def __init__(self, plot_item, selection: SelectionState) -> None:
        self.selection = selection
        self._plot = plot_item
        self._syncing = False

        if not PG_OK or plot_item is None:
            self._line = None
            self._region = None
            return

        self._line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen((90, 200, 120), width=1)
        )
        self._line.setZValue(10)
        self._line.sigPositionChanged.connect(self._on_line_moved)
        plot_item.addItem(self._line, ignoreBounds=True)

        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(90, 200, 120, 40), pen=pg.mkPen((90, 200, 120), style=Qt.PenStyle.DotLine)
        )
        self._region.setZValue(5)
        self._region.setVisible(False)
        self._region.sigRegionChangeFinished.connect(self._on_region_moved)
        plot_item.addItem(self._region, ignoreBounds=True)

        selection.changed.connect(self._on_selection_changed)
        self._apply_from_selection()

    # -- selection -> widgets --------------------------------------
    def _on_selection_changed(self, field: str) -> None:
        if field in ("timestep", "time_range", "*"):
            self._apply_from_selection()

    def _apply_from_selection(self) -> None:
        if self._line is None or self._syncing:
            return
        self._syncing = True
        try:
            ts = self.selection.get("timestep")
            if ts is not None:
                self._line.setPos(float(ts))
            tr = self.selection.get("time_range")
            if tr and len(tr) == 2:
                self._region.setRegion((float(tr[0]), float(tr[1])))
                self._region.setVisible(True)
            else:
                self._region.setVisible(False)
        finally:
            self._syncing = False

    # -- widgets -> selection --------------------------------------
    def _on_line_moved(self) -> None:
        if self._syncing or self._line is None:
            return
        self._syncing = True
        try:
            self.selection.set("timestep", int(round(self._line.value())), cascade=False)
        finally:
            self._syncing = False

    def _on_region_moved(self) -> None:
        if self._syncing or self._region is None:
            return
        self._syncing = True
        try:
            lo, hi = self._region.getRegion()
            self.selection.set("time_range", (int(round(lo)), int(round(hi))), cascade=False)
        finally:
            self._syncing = False

    def enable_region(self, enabled: bool) -> None:
        if self._region is not None:
            self._region.setVisible(enabled)

    def rebind(self, plot_item) -> None:
        """Move the cursor items onto a freshly-created ``PlotItem``.

        Views that rebuild their plots on every render (``GraphicsLayoutWidget``
        + ``clear()``) call this instead of constructing a new ``TimeCursor``,
        so the ``selection.changed`` subscription is made once, not leaked per
        render.
        """
        if self._line is None or plot_item is None or plot_item is self._plot:
            self._apply_from_selection()
            return
        for item in (self._line, self._region):
            try:
                self._plot.removeItem(item)
            except Exception:  # noqa: BLE001 - old plot already gone
                pass
        self._plot = plot_item
        plot_item.addItem(self._line, ignoreBounds=True)
        plot_item.addItem(self._region, ignoreBounds=True)
        self._apply_from_selection()

    def reattach(self) -> None:
        """Re-add the cursor items after a ``PlotItem.clear()`` wiped them."""
        if self._line is None:
            return
        items = self._plot.items if hasattr(self._plot, "items") else []
        if self._line not in items:
            self._plot.addItem(self._line, ignoreBounds=True)
        if self._region not in items:
            self._plot.addItem(self._region, ignoreBounds=True)
        self._apply_from_selection()
