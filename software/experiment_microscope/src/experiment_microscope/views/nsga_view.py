"""NSGA-II / paraconsistentGA view (FIXME §48).

    D_penalized
          ↑
          │      ○        ○  = evaluated individual
          │   ○ ●         ●  = feasible Pareto front
          │  ●   ○
          └──────────────→  inference cost

One run at a time. The feasible Pareto front is drawn filled; the rest of the
evaluated population is hollow. Selecting an individual shows its genome,
fitness, feasibility and (UNCALIBRATED) estimated latency. Clicking emits the
individual so the paraconsistent / reconstruction views can follow.

The repository marks estimated latency as **UNCALIBRATED**; this view repeats
that on every latency value and never ranks by it silently.
"""

from __future__ import annotations

from experiment_microscope.views._help import HelpBox

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views._pg import PG_OK, missing_widget, pg

_X_KEYS = ("inference_cost", "param_count", "est_latency_ms", "latent_activity")


_HELP = """
<b>What this shows.</b> The result of <b>NSGA-II</b> (Non-dominated Sorting
Genetic Algorithm II), a multi-objective architecture search: it evolves a
population and keeps the candidates that are not beaten on every objective at
once.
<br><br>
<b>Axes.</b> Vertical = D_penalized mean (paraconsistent quality, smaller
better). Horizontal = the cost you pick: <b>inference cost</b> (spikes + 10 ×
multiply-accumulates, hardware-independent), parameter count, estimated latency,
or latent activity.
<br><br>
<b>Green</b> = <b>feasible</b> Pareto-front points (satisfy every hard
constraint); <b>orange</b> = infeasible — shown for context, never winners.
Estimated latency is labelled <b>UNCALIBRATED</b>: it is a rough model output,
not a measured millisecond figure — the view never ranks by it. Click a point
for its genome and fitness.
"""


class NsgaView(QWidget):
    individual_clicked = Signal(object)  # dict

    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._adapter = repo.adapter("paraconsistent_ga")
        self._pop: dict = {}

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('NSGA-II search', _HELP))
        bar = QHBoxLayout()
        self._run = QComboBox()
        self._run.currentIndexChanged.connect(self._load)
        self._x = QComboBox()
        self._x.addItems(_X_KEYS)
        self._x.currentIndexChanged.connect(self._replot)
        bar.addWidget(QLabel("run"))
        bar.addWidget(self._run, 1)
        bar.addWidget(QLabel("x axis"))
        bar.addWidget(self._x)
        root.addLayout(bar)
        self._status = QLabel("—")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        if not PG_OK:
            root.addWidget(missing_widget("NSGA-II view"))
            self._plot = None
            return
        split = QSplitter(Qt.Orientation.Horizontal)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "D_penalized  (mean over seeds)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._scatter = pg.ScatterPlotItem(size=9, hoverable=True, tip=lambda x, y, d: _tip(d))
        self._scatter.sigClicked.connect(self._on_click)
        self._plot.addItem(self._scatter)
        split.addWidget(self._plot)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["field", "value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        split.addWidget(self._table)
        split.setSizes([560, 320])
        root.addWidget(split, 1)
        self.refresh()

    def show_node(self, node, adapter_key: str) -> None:
        if adapter_key == "paraconsistent_ga":
            h = getattr(node, "handle", {}) or {}
            if h.get("run_tag"):
                i = self._run.findText(h["run_tag"])
                if i >= 0:
                    self._run.setCurrentIndex(i)
            if h.get("level") == "individual" and h.get("index") is not None:
                front = self._pop.get("front") or []
                if 0 <= h["index"] < len(front):
                    self._show_individual(front[h["index"]])

    def refresh(self) -> None:
        runs = self._adapter.ga_runs()
        cur = self._run.currentText()
        self._run.blockSignals(True)
        self._run.clear()
        self._run.addItems(runs)
        if cur in runs:
            self._run.setCurrentText(cur)
        self._run.blockSignals(False)
        if runs:
            self._load()
        else:
            self._status.setText("No paraconsistentGA results on disk.")

    def _load(self) -> None:
        tag = self._run.currentText()
        if not tag:
            return
        self._pop = self._adapter.population(tag)
        warn = self._pop.get("warnings") or []
        ga = self._pop.get("ga") or {}
        n_eval = self._pop.get("n_evaluated")
        self._status.setText(
            f"{tag} — {len(self._pop['front'])} Pareto-front individual(s); "
            f"GA: pop {ga.get('population_size', '?')} × {ga.get('generations', '?')} gen, "
            f"{n_eval if n_eval is not None else '?'} evaluated"
            + (f"  ·  {'; '.join(map(str, warn))}" if warn else "")
        )
        self._replot()

    def _replot(self) -> None:
        if self._plot is None:
            return
        xk = self._x.currentText()
        self._plot.setLabel("bottom", xk + ("  [UNCALIBRATED]" if xk == "est_latency_ms" else ""))
        spots = []
        for ind in self._pop.get("front", []):
            x, y = ind.get(xk), ind.get("d_penalized_mean")
            if x is None or y is None:
                continue
            feasible = bool(ind.get("feasible"))
            spots.append({
                "pos": (float(x), float(y)), "data": ind,
                "brush": pg.mkBrush(90, 220, 140, 230) if feasible else pg.mkBrush(200, 130, 90, 160),
                "pen": pg.mkPen((40, 140, 90)) if feasible else pg.mkPen((160, 90, 50)),
                "size": 11 if feasible else 8,
            })
        self._scatter.setData(spots)
        from experiment_microscope.views._plotinfo import autofit, set_source

        set_source(self._plot, "results/paraconsistentGA/pga_*_pareto.json (Pareto front) · "
                   "hover a point for genome + fitness")
        autofit(self._plot)  # every point of the front in frame on each replot

    def _on_click(self, _s, points) -> None:
        if not len(points):
            return
        ind = points[0].data()
        self._show_individual(ind)
        self.individual_clicked.emit(ind)

    def _show_individual(self, ind: dict) -> None:
        genome = ind.get("genome") or {}
        rows: list[tuple[str, str]] = [
            ("rank", str(ind.get("rank", "—"))),
            ("feasible", str(ind.get("feasible", "—"))),
            ("born_generation", str(ind.get("born_generation", "—"))),
            ("D_truth", _f(ind.get("d_truth"))),
            ("D_penalized (mean)", _f(ind.get("d_penalized_mean"))),
            ("D_penalized (std)", _f(ind.get("d_penalized_std"))),
            ("alpha", _f(ind.get("alpha"))),
            ("beta", _f(ind.get("beta"))),
            ("g1", _f(ind.get("g1"))),
            ("g2", _f(ind.get("g2"))),
            ("inference_cost", str(ind.get("inference_cost", "—"))),
            ("param_count", str(ind.get("param_count", "—"))),
            ("est_latency_ms", f"{_f(ind.get('est_latency_ms'))}  (UNCALIBRATED)"),
            ("latent_activity", _f(ind.get("latent_activity"))),
            ("constraint_violation", _f(ind.get("constraint_violation"))),
        ]
        rows += [(f"genome.{k}", str(v)) for k, v in genome.items()]
        self._table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(k))
            self._table.setItem(r, 1, QTableWidgetItem(v))


def _f(x) -> str:
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return "—"


def _tip(ind) -> str:
    if not isinstance(ind, dict):
        return ""
    g = ind.get("genome") or {}
    return (f"rank {ind.get('rank', '?')}  feasible={ind.get('feasible')}\n"
            f"D_penalized={_f(ind.get('d_penalized_mean'))}  D_truth={_f(ind.get('d_truth'))}\n"
            f"cost={ind.get('inference_cost')}  params={ind.get('param_count')}\n"
            f"latent={g.get('latent')}  depth={g.get('depth')}  enc={g.get('encoding')}\n"
            f"est_latency_ms={_f(ind.get('est_latency_ms'))} (UNCALIBRATED)")
