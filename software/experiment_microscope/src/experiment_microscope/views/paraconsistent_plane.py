"""Paraconsistent plane dock (FIXME §12).

Scatter of every persisted feature-set score on the G1×G2 plane (certainty ×
contradiction). Convention from ``ThesisParaconsistent.hpp``:

    g1 = alpha - beta          (certainty degree; +1 = Truth, -1 = False)
    g2 = alpha + beta - 1      (contradiction degree; +1 = Inconsistent, -1 = Indeterminate)
    d_truth      = distance to the Truth vertex (g1=1, g2=0)
    d_penalized  = d_truth + (2 - sqrt(2)) * |g2|     <-- ranking metric, ascending

Hover a point for all six quantities. Clicking sets the shared selection so the
feature-matrix / reconstruction views can follow.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from experiment_microscope.data.adapters import ParaconsistentPoint
from experiment_microscope.views._help import HelpBox
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views._pg import PG_OK, missing_widget, pg

_K = 0.5857864376269049  # 2 - sqrt(2), kContradictionPenalty


_HELP = """
<b>What this shows.</b> Each feature set placed on the <b>paraconsistent</b>
plane. Paraconsistent logic lets a claim be supported <i>and</i> denied at once,
which is exactly what noisy biometric evidence looks like.
<br><br>
<b>Axes.</b> Horizontal <b>G1 = α − β</b> (certainty): +1 = evidence firmly says
"same person", −1 = firmly "different", 0 = undecided. Vertical
<b>G2 = α + β − 1</b> (contradiction): +1 = evidence fully conflicts with itself,
−1 = evidence missing, 0 = clean. α (alpha) is evidence <i>for</i>, β (beta) is
evidence <i>against</i>, both 0…1.
<br><br>
The ideal corner is <b>(G1 = 1, G2 = 0)</b> — certainly true, no contradiction.
<b>D_truth</b> is the straight-line distance to it; <b>D_penalized</b> adds
(2 − √2)·|G2| for contradiction and is the number the ranking sorts on. Smaller
is better. Hover a point for all six quantities.
"""



class ParaconsistentPlane(QWidget):
    point_clicked = Signal(object)  # ParaconsistentPoint

    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._points: list[ParaconsistentPoint] = []
        layout = QVBoxLayout(self)
        layout.addWidget(HelpBox('Paraconsistent plane', _HELP))
        layout.setContentsMargins(0, 0, 0, 0)
        if not PG_OK:
            layout.addWidget(missing_widget("Paraconsistent plane"))
            self._plot = None
            return
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "G1  (certainty:  alpha - beta)")
        self._plot.setLabel("left", "G2  (contradiction:  alpha + beta - 1)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setXRange(-1.05, 1.05)
        self._plot.setYRange(-1.05, 1.05)
        _axis_pen = pg.mkPen((120, 120, 120), style=Qt.PenStyle.DashLine)
        self._plot.addLine(x=0, pen=_axis_pen)
        self._plot.addLine(y=0, pen=_axis_pen)
        for x, y, txt in ((1, 0, "Truth"), (-1, 0, "False"),
                          (0, 1, "Inconsistent"), (0, -1, "Indeterminate")):
            label = pg.TextItem(txt, color=(160, 160, 160), anchor=(0.5, 0.5))
            label.setPos(x * 0.92, y * 0.92)
            self._plot.addItem(label)
        self._scatter = pg.ScatterPlotItem(
            size=9,
            pen=pg.mkPen(None),
            hoverable=True,
            tip=lambda x, y, data: _tooltip(data) if isinstance(data, ParaconsistentPoint) else f"({x:.3g}, {y:.3g})",
        )
        self._scatter.sigClicked.connect(self._on_click)
        self._plot.addItem(self._scatter)
        layout.addWidget(self._plot)

    def refresh(self) -> None:
        if self._plot is None:
            return
        self._points = self.repo.all_paraconsistent_points()
        spots = []
        for p in self._points:
            if p.g1.is_missing or p.g2.is_missing:
                continue
            dp = None if p.d_penalized.is_missing else float(p.d_penalized.magnitude)
            spots.append(
                {
                    "pos": (float(p.g1.magnitude), float(p.g2.magnitude)),
                    "data": p,
                    "brush": _brush_for(dp),
                }
            )
        self._scatter.setData(spots)

    def can_export(self) -> bool:
        return bool(self._points)

    def export_figure(self, path, **opts):
        import numpy as np

        from experiment_microscope.viz.mpl_export import annotate_provenance, new_figure, save_figure

        fig = new_figure(width_in=opts.get("width_in", 4.5), height_in=opts.get("height_in", 4.5))
        ax = fig.add_subplot(111)
        xs, ys = [], []
        for p in self._points:
            if p.g1.is_missing or p.g2.is_missing:
                continue
            xs.append(float(p.g1.magnitude))
            ys.append(float(p.g2.magnitude))
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="0.6", lw=0.6)
        ax.scatter(xs, ys, s=14, alpha=0.75)
        for x, y, t in ((1, 0, "Truth"), (-1, 0, "False"), (0, 1, "Inconsistent"),
                        (0, -1, "Indeterminate")):
            ax.annotate(t, (x * 0.9, y * 0.9), fontsize=6, color="0.5", ha="center")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("G1  (alpha - beta)")
        ax.set_ylabel("G2  (alpha + beta - 1)")
        ax.set_aspect("equal")
        annotate_provenance(ax, f"origin: measured (persisted *_paraconsistent.csv); {len(xs)} sets")
        return save_figure(fig, path, transparent=opts.get("transparent", False))

    def _on_click(self, _scatter, points) -> None:
        if not len(points):
            return
        payload = points[0].data()
        if isinstance(payload, ParaconsistentPoint):
            self.point_clicked.emit(payload)


def _brush_for(d_penalized: float | None):
    if d_penalized is None:
        return pg.mkBrush(150, 150, 150, 200)
    # lower d_penalized ranks higher; map to a simple green->red ramp
    t = max(0.0, min(1.0, d_penalized / 2.0))
    return pg.mkBrush(int(255 * t), int(200 * (1 - t)), 80, 220)


def _tooltip(p: ParaconsistentPoint) -> str:
    rows = [p.label]
    for name, val in (("alpha", p.alpha), ("beta", p.beta), ("G1", p.g1),
                      ("G2", p.g2), ("D_truth", p.d_truth), ("D_penalized", p.d_penalized)):
        rows.append(f"{name}: {val.labelled()}")
    for k, v in (p.facet or {}).items():
        rows.append(f"{k}: {getattr(v, 'display', lambda: v)() if hasattr(v, 'display') else v}")
    return "\n".join(rows)
