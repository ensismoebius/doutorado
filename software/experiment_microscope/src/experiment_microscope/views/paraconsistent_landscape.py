"""Paraconsistent feature landscape (FIXME §13).

A second view of the same scores the G1×G2 plane shows, arranged to make the
*ranking penalty* visible:

    X = D_truth       (distance of the feature set to the Truth vertex)
    Y = D_penalized   = D_truth + (2 − √2)·|g2|      (the ascending rank metric)

Every point sits on or above the ``y = x`` line; how far above is exactly the
contradiction penalty ``(2 − √2)·|g2|``. Filters (dataset / modality / wavelet /
scale / fold / seed / strategy) narrow the cloud. Clicking a point emits it so
the feature-matrix / triangle views can inspect the underlying vectors.
"""

from __future__ import annotations

from experiment_microscope.views._help import HelpBox

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.data.adapters import ParaconsistentPoint
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views._pg import PG_OK, missing_widget, pg

_ALL = "(all)"
_FILTERS = ("dataset", "modality", "wavelet", "scale", "fold", "seed", "strategy")
_TAG_RE = re.compile(r"hc_(?P<wavelet>daub\d+|haar)_(?P<scale>[a-z]+)_c(?P<cat>\d)_(?P<modality>[a-z]+)")


def _facets(p: ParaconsistentPoint) -> dict[str, str]:
    f = dict(p.facet or {})
    tag = str(f.get("run_tag", ""))
    m = _TAG_RE.search(tag)
    out = {
        "dataset": str(f.get("dataset") or f.get("modality") or ""),
        "modality": str(f.get("modality") or (m["modality"] if m else "")),
        "wavelet": m["wavelet"] if m else "",
        "scale": m["scale"] if m else "",
        "fold": str(f.get("fold") if f.get("fold") is not None else ""),
        "seed": str(_scalar(f.get("seed"))),
        "strategy": str(f.get("strategy") or ("handcrafted" if tag.startswith(("e05_p00_hc", "hc")) else "")),
    }
    return out


def _scalar(v):
    return getattr(v, "magnitude", v) if v is not None else ""


_HELP = """
<b>What this shows.</b> Every persisted score as a point at
(<b>D_truth</b>, <b>D_penalized</b>). The dashed diagonal is D_penalized =
D_truth; the vertical gap above it <i>is</i> the contradiction penalty
(2 − √2)·|G2|, so points far above the line have self-conflicting evidence.
Point colour scales with that gap.
<br><br>
D_truth = distance on the (G1, G2) plane to the ideal "certainly true, no
contradiction" corner. D_penalized = D_truth + the penalty; it is what the
ranking sorts on. Both: smaller is better.
<br><br>
The seven facet filters (dataset, modality, wavelet, scale, fold, seed,
strategy) narrow the cloud. Click a point to inspect it.
"""


class ParaconsistentLandscape(QWidget):
    point_clicked = Signal(object)

    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._points: list[ParaconsistentPoint] = []

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Paraconsistent landscape', _HELP))
        bar = QHBoxLayout()
        form = QFormLayout()
        self._combos: dict[str, QComboBox] = {}
        for name in _FILTERS:
            c = QComboBox()
            c.addItem(_ALL)
            c.currentIndexChanged.connect(self._replot)
            form.addRow(name, c)
            self._combos[name] = c
        bar.addLayout(form)
        self._status = QLabel("—")
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        if not PG_OK:
            root.addWidget(missing_widget("Paraconsistent landscape"))
            self._plot = None
            return
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "D_truth  (distance to Truth vertex)")
        self._plot.setLabel("left", "D_penalized  = D_truth + (2−√2)·|g2|")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._ref = self._plot.plot([], [], pen=pg.mkPen((120, 120, 120), style=Qt.PenStyle.DashLine))
        self._scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), hoverable=True,
                                           tip=lambda x, y, data: _tip(data))
        self._scatter.sigClicked.connect(self._on_click)
        self._plot.addItem(self._scatter)
        root.addWidget(self._plot, 1)
        self.refresh()

    def show_node(self, node, adapter_key: str) -> None:
        """Landscape is pipeline-wide; selection does not change it."""

    def refresh(self) -> None:
        self._points = self.repo.all_paraconsistent_points()
        for name, combo in self._combos.items():
            vals = sorted({_facets(p)[name] for p in self._points if _facets(p)[name]})
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_ALL)
            combo.addItems(vals)
            combo.setCurrentText(cur if cur in vals else _ALL)
            combo.blockSignals(False)
        self._replot()

    def _selected(self) -> list[ParaconsistentPoint]:
        want = {n: c.currentText() for n, c in self._combos.items() if c.currentText() != _ALL}
        out = []
        for p in self._points:
            fc = _facets(p)
            if all(fc.get(k) == v for k, v in want.items()):
                out.append(p)
        return out

    def _replot(self) -> None:
        if self._plot is None:
            return
        pts = self._selected()
        spots, xs = [], []
        for p in pts:
            if p.d_truth.is_missing or p.d_penalized.is_missing:
                continue
            dt, dp = float(p.d_truth.magnitude), float(p.d_penalized.magnitude)
            xs.append(dt)
            pen_amt = max(0.0, dp - dt)
            spots.append({"pos": (dt, dp), "data": p,
                          "brush": pg.mkBrush(int(min(255, 60 + pen_amt * 400)), 120, 200, 210)})
        self._scatter.setData(spots)
        from experiment_microscope.views._plotinfo import autofit, set_source

        set_source(self._plot, "persisted *_paraconsistent.csv · [measured] · "
                   "hover a point for its facets + D values")
        if xs:
            lo, hi = min(xs), max(xs)
            self._ref.setData([lo, hi], [lo, hi])
        autofit(self._plot)  # re-fit to the filtered cloud on every facet change
        self._status.setText(
            f"{len(spots)} / {len(self._points)} feature set(s) shown — "
            f"vertical gap above y=x is the contradiction penalty (2−√2)·|g2|"
        )

    def _on_click(self, _scatter, points) -> None:
        if len(points):
            payload = points[0].data()
            if isinstance(payload, ParaconsistentPoint):
                self.point_clicked.emit(payload)


def _tip(p) -> str:
    if not isinstance(p, ParaconsistentPoint):
        return ""
    fc = _facets(p)
    head = [p.label]
    head += [f"{k}: {v}" for k, v in fc.items() if v]
    head += [f"D_truth: {p.d_truth.labelled()}", f"D_penalized: {p.d_penalized.labelled()}"]
    return "\n".join(head)
