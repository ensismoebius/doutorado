"""Latent Space Explorer (FIXME §19).

For a meeting01 fold: run every test/val/train window of that fold through the
trained SNN-AE, collect the latent vectors, and project them to 2-D (PCA or
t-SNE) — or 3-D PCA in the embedded VTK panel. Points are coloured by digit or
speaker. Clicking a point selects that window everywhere else in the app
(``sample_activated``), so the explorer is a launch pad back into the
raw-signal / wavelet / reconstruction chain, not an isolated picture.

The projection is tagged ``PROJECTED`` — it is a lossy view of a 32-D vector,
never a pipeline number (FIXME §33). The forward passes run on a worker thread
so the GUI never blocks (FIXME §32).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.integrity import Origin
from experiment_microscope.views._help import HelpBox
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.viz.pyvista_panel import PV_OK, PyVistaPanel, pv

_SPLITS = ("test", "val", "train")
_METHODS = ("PCA", "t-SNE")
_COLOR_BY = ("digit", "speaker")


_HELP = """
<b>What this shows.</b> Every window of one meeting01 <b>fold</b> pushed through
the trained autoencoder, then its 32-number <b>latent</b> vector squeezed to 2-D
(or 3-D) so you can see the structure.
<br><br>
<b>This is a PROJECTED view</b>, not a pipeline number.
<b>PCA</b> (Principal Component Analysis) rotates onto the directions of greatest
spread — distances stay roughly meaningful, and the title shows how much
variability the 2 axes keep. <b>t-SNE</b> only preserves <i>who is near whom</i>;
gaps and cluster sizes are not meaningful.
<br><br>
<b>Colour</b> = digit spoken, or speaker (legend/​controls). Well-separated
colours mean the latent space encodes that property. <b>Click a point</b> to
select that window everywhere else in the app.
<br><br>
The forward passes run on a background thread — press <i>Project</i> and wait for
the status line.
"""



class _BatchWorker(QObject):
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, adapter, dataset, fold, split, limit) -> None:
        super().__init__()
        self._args = (adapter, dataset, fold, split, limit)

    def run(self) -> None:
        adapter, dataset, fold, split, limit = self._args
        try:
            batch = adapter.latent_batch(dataset, fold, split=split, limit=limit)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.done.emit(batch)


class LatentExplorer(QWidget):
    sample_activated = Signal(str, int, str, int)  # dataset, fold, split, row
    working = Signal(bool)  # True when the worker thread is running

    def __init__(self, repo, app_state=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.app_state = app_state
        self._ds: str | None = None
        self._fold: int | None = None
        self._batch: dict | None = None
        self._coords: np.ndarray | None = None
        self._thread: QThread | None = None

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Latent Space Explorer', _HELP))
        bar = QHBoxLayout()
        self._split = QComboBox()
        self._split.addItems(_SPLITS)
        self._limit = QSpinBox()
        self._limit.setRange(8, 500)
        self._limit.setValue(120)
        self._method = QComboBox()
        self._method.addItems(_METHODS)
        self._color = QComboBox()
        self._color.addItems(_COLOR_BY)
        self._dims = QComboBox()
        self._dims.addItems(("2-D", "3-D (PCA)"))
        self._go = QPushButton("Project")
        self._go.clicked.connect(self._recompute)
        for w in (self._method, self._color, self._dims):
            w.currentIndexChanged.connect(self._render)
        for lbl, w in (("split", self._split), ("max windows", self._limit),
                       ("method", self._method), ("colour", self._color), ("", self._dims)):
            if lbl:
                bar.addWidget(QLabel(lbl))
            bar.addWidget(w)
        bar.addWidget(self._go)
        self._status = QLabel("Select a meeting01 fold or window, then Project.")
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        if PG_OK:
            self._plot = pg.PlotWidget()
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("bottom", "projection axis 1 (arbitrary units — a direction, not a measurement)")
            self._plot.setLabel("left", "projection axis 2 (arbitrary units)")
            self._scatter = pg.ScatterPlotItem(
                size=9, pen=pg.mkPen(None), hoverable=True, tip=self._point_tip
            )
            self._scatter.sigClicked.connect(self._on_point_clicked)
            self._plot.addItem(self._scatter)
            root.addWidget(self._plot, 1)
        else:
            self._plot = None
            root.addWidget(missing_widget("Latent Space Explorer"))
        self._panel = PyVistaPanel()
        self._panel.setVisible(False)
        root.addWidget(self._panel, 1)

    # -- external API --------------------------------------------------
    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        h = getattr(node, "handle", {}) or {}
        if adapter_key != "meeting01" or h.get("dataset") is None or h.get("cv_fold") is None:
            self._status.setText("Latent Space Explorer needs a meeting01 fold / window.")
            return
        ds, fold = h["dataset"], h["cv_fold"]
        if (ds, fold) != (self._ds, self._fold):
            self._ds, self._fold, self._batch, self._coords = ds, fold, None, None
            self._status.setText(f"meeting01 › {ds} › fold {fold} — press Project.")

    # -- compute ------------------------------------------------------
    def _recompute(self) -> None:
        if self._ds is None or self._fold is None:
            self._status.setText("Select a meeting01 fold first.")
            return
        if self._thread is not None:
            return
        try:
            adapter = self.repo.adapter("meeting01")
        except BindingUnavailableError as exc:
            self._status.setText(str(exc))
            return
        self._go.setEnabled(False)
        self._status.setText("running SNN-AE forward passes on a worker thread…")
        self.working.emit(True)
        self._thread = QThread(self)
        self._worker = _BatchWorker(
            adapter, self._ds, self._fold, self._split.currentText(), self._limit.value()
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_batch)
        self._worker.failed.connect(self._on_fail)
        self._thread.start()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._go.setEnabled(True)
        self.working.emit(False)

    def _on_fail(self, msg: str) -> None:
        self._teardown_thread()
        self._status.setText(msg)

    def _on_batch(self, batch: dict) -> None:
        self._teardown_thread()
        self._batch = batch
        n = batch["latents"].shape[0]
        spec = batch["spec"]
        note = "" if batch["lif_params"] else "  — LIF params default (pre-fix checkpoint)"
        self._status.setText(
            f"{n} latents ({spec['architecture']}/{spec['encoding']}, dim "
            f"{batch['latents'].shape[1]}) — projection is PROJECTED, not a pipeline value{note}"
        )
        self._render()

    # -- render -----------------------------------------------------
    def _project(self, k: int) -> tuple[np.ndarray, str]:
        from experiment_microscope.processing import projections as pj

        m = self._batch["latents"]
        if self._method.currentText() == "PCA" or m.shape[0] < 8:
            coords, evr = pj.pca(m, n_components=k)
            tail = "  var " + "/".join(f"{v:.0%}" for v in evr[:k])
            return coords, f"PCA{tail}"
        coords = pj.tsne(m, n_components=k, seed=0)
        return coords, "t-SNE"

    def _colours(self) -> tuple[list, dict, dict]:
        key = "digits" if self._color.currentText() == "digit" else "speakers"
        vals = self._batch[key]
        uniq = sorted({v for v in vals if v is not None}, key=lambda x: str(x))
        idx = {v: i for i, v in enumerate(uniq)}
        lut = {v: pg.intColor(i, hues=max(3, len(uniq))) for v, i in idx.items()}
        return vals, lut, idx

    def _render(self) -> None:
        if self._batch is None:
            return
        want_3d = self._dims.currentText().startswith("3")
        k = 3 if want_3d else 2
        try:
            coords, label = self._project(k)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"projection failed: {exc}")
            return
        self._coords = coords
        vals, lut, idx = self._colours()

        if want_3d and PV_OK and self._panel.available:
            self._panel.setVisible(True)
            if self._plot is not None:
                self._plot.setVisible(False)
            p = self._panel.plotter
            p.clear()
            cloud = pv.PolyData(coords[:, :3].astype(float))
            cloud["class"] = np.array([idx.get(v, -1) for v in vals], dtype=float)
            p.add_mesh(cloud, scalars="class", cmap="tab20", render_points_as_spheres=True,
                       point_size=12, show_scalar_bar=False)
            try:
                p.reset_camera()
            except Exception:  # noqa: BLE001
                pass
            self._plot_title(label)
            return

        self._panel.setVisible(False)
        if self._plot is None:
            return
        self._plot.setVisible(True)
        spots = []
        for i, (x, y) in enumerate(coords[:, :2]):
            c = lut.get(vals[i], pg.mkColor(150, 150, 150))
            spots.append({"pos": (float(x), float(y)), "data": i, "brush": c,
                          "symbol": "o", "size": 9})
        self._scatter.setData(spots)
        self._legend(lut, idx)
        from experiment_microscope.views._plotinfo import set_source

        set_source(self._plot, "meeting01.latent_batch() → SNN-AE latents, "
                   f"projected with {label.split()[0]} · PROJECTED (hover a point)")
        self._plot_title(label)

    def _legend(self, lut: dict, idx: dict) -> None:
        """A real colour key: one entry per class value."""
        if self._plot is None:
            return
        pi = self._plot.getPlotItem()
        if pi.legend is not None:
            pi.legend.scene().removeItem(pi.legend)
            pi.legend = None
        leg = pi.addLegend(offset=(-10, 10))
        c = self._color.currentText()
        for val in sorted(idx, key=lambda x: str(x)):
            dot = pg.ScatterPlotItem([0], [0], symbol="o", size=9, brush=lut[val],
                                     pen=pg.mkPen(None))
            leg.addItem(dot, f"{c} {val}")

    def _point_tip(self, x: float, y: float, data) -> str:
        b = self._batch
        if b is None or not isinstance(data, int):
            return f"({x:.3g}, {y:.3g})"
        return (
            f"window row {b['rows'][data]} ({b['split']})\n"
            f"digit {b['digits'][data]}   speaker {b['speakers'][data]}\n"
            f"projected coords ({x:.3g}, {y:.3g}) — not a pipeline value"
        )

    def _plot_title(self, method_label: str) -> None:
        c = self._color.currentText()
        if self._plot is not None:
            self._plot.setTitle(
                f"latent vectors → {method_label}   [projected — a view, not a "
                f"measurement]   ·   colour = {c}")

    def _on_point_clicked(self, _scatter, points) -> None:
        if not points or self._batch is None:
            return
        i = int(points[0].data())
        row = self._batch["rows"][i]
        split = self._batch["split"]
        self._status.setText(
            f"window row {row} ({split}) — digit {self._batch['digits'][i]}, "
            f"speaker {self._batch['speakers'][i]}  → selected everywhere"
        )
        self.sample_activated.emit(self._ds, int(self._fold), split, row)

    # provenance origin for any caption
    origin = Origin.PROJECTED
