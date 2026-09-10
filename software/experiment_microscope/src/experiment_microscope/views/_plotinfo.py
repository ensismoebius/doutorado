"""Plot provenance + hover readout (FIXME §29, §55).

Two things every data plot should have:

* ``set_source`` — a dim one-line caption *on the plot* saying where the numbers
  came from (which adapter call / file / origin tag), so the plot is
  self-describing without reading the surrounding prose.
* ``HoverReadout`` — move the mouse over a line plot and a label follows the
  cursor showing the x position and every curve's y value there, plus a thin
  vertical guide. Works for any pyqtgraph ``PlotItem`` with ``plot()`` curves,
  no per-view code. Scatter plots use pyqtgraph's own ``hoverable=True`` /
  ``tip=`` instead (a per-point tooltip).
"""

from __future__ import annotations

import numpy as np

from experiment_microscope.views._pg import PG_OK, pg


def set_source(plot, text: str) -> None:
    """Put a small grey provenance caption in the plot's top-left corner.

    Re-callable: the previous caption is replaced. ``plot`` may be a
    ``PlotWidget`` or a ``PlotItem``.
    """
    if not PG_OK or not text:
        return
    try:
        pi = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
        vb = pi.getViewBox()
        old = getattr(pi, "_source_label", None)
        if old is not None:
            try:
                vb.removeItem(old)
            except Exception:  # noqa: BLE001
                pass
        lbl = pg.TextItem(color=(150, 150, 150), anchor=(0, 0))
        lbl.setHtml(f"<div style='font-size:8pt;color:#999'>source: {text}</div>")
        lbl.setParentItem(vb)
        lbl.setPos(4, 4)  # pixels from the top-left of the view, not data coords
        pi._source_label = lbl
    except Exception:  # noqa: BLE001 - provenance caption must never break a view
        pass


class HoverReadout:
    """Attach a cursor-following x/y readout to a pyqtgraph ``PlotItem``.

    Keep a reference (the ``SignalProxy`` dies with it). For views that rebuild
    their plots every render, just construct a fresh one each time — the old one
    is garbage-collected with its plot.
    """

    def __init__(self, plot, *, x_label: str = "x", precision: int = 4) -> None:
        self._ok = False
        if not PG_OK:
            return
        try:
            self._pi = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
        except Exception:  # noqa: BLE001
            return
        self._x_label = x_label
        self._prec = precision
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen((160, 160, 160, 120), width=1))
        self._vline.setZValue(50)
        self._pi.addItem(self._vline, ignoreBounds=True)
        self._text = pg.TextItem(color=(230, 230, 230), anchor=(0, 1),
                                 fill=pg.mkBrush(30, 30, 30, 210))
        self._text.setZValue(51)
        self._pi.addItem(self._text, ignoreBounds=True)
        self._vline.hide()
        self._text.hide()
        self._proxy = pg.SignalProxy(self._pi.scene().sigMouseMoved,
                                     rateLimit=60, slot=self._on_move)
        self._ok = True

    def reattach(self) -> None:
        """Re-add the guide + label after a ``PlotItem.clear()`` (which drops
        every added item). Call right after re-plotting, like ``TimeCursor``."""
        if not self._ok:
            return
        for it in (self._vline, self._text):
            if it not in self._pi.items:
                self._pi.addItem(it, ignoreBounds=True)
        self._vline.hide()
        self._text.hide()

    def _on_move(self, evt) -> None:
        if not self._ok:
            return
        pos = evt[0]
        vb = self._pi.getViewBox()
        if not self._pi.sceneBoundingRect().contains(pos):
            self._vline.hide()
            self._text.hide()
            return
        mp = vb.mapSceneToView(pos)
        x = float(mp.x())
        rows = []
        for item in self._pi.listDataItems():
            xd, yd = item.getData()
            if xd is None or len(xd) == 0:
                continue
            xd = np.asarray(xd)
            yd = np.asarray(yd)
            j = int(np.clip(np.searchsorted(xd, x), 0, len(xd) - 1))
            if j > 0 and abs(xd[j - 1] - x) < abs(xd[j] - x):
                j -= 1
            name = item.name() or "value"
            rows.append(f"{name}: {yd[j]:.{self._prec}g}")
        if not rows:
            self._vline.hide()
            self._text.hide()
            return
        self._vline.setPos(x)
        self._vline.show()
        self._text.setText(f"{self._x_label} = {x:.{self._prec}g}\n" + "\n".join(rows))
        self._text.setPos(x, mp.y())
        self._text.show()
