"""Reconstruction view (FIXME §20).

For one window pushed through a trained SNN autoencoder:

    original (encoder input)   ── the flattened, encoded window
    reconstruction             ── decoder output, same length
    residual                   ── original − reconstruction

plus MSE / MAE / R² / Pearson r. Needs a trained ``*_encoder.npz`` /
``*_decoder.npz`` (Step E); until a LOSO fold has written one, the panel shows
the exact remedy rather than a fabricated curve (no-fallback, §33).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg


class ReconstructionView(QWidget):
    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._trace = None

        root = QVBoxLayout(self)
        self._status = QLabel("Select a meeting01 window.")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        if not PG_OK:
            root.addWidget(missing_widget("Reconstruction view"))
            self._plot = None
            return
        split = QSplitter(Qt.Orientation.Vertical)
        self._layout_widget = pg.GraphicsLayoutWidget()
        split.addWidget(self._layout_widget)
        self._metrics = QTableWidget(0, 2)
        self._metrics.setHorizontalHeaderLabels(["metric", "value"])
        self._metrics.horizontalHeader().setStretchLastSection(True)
        split.addWidget(self._metrics)
        split.setSizes([440, 150])
        root.addWidget(split, 1)
        self._plot = self._layout_widget

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._plot is None:
            return
        if adapter_key != "meeting01" or (getattr(node, "handle", {}) or {}).get("level") != "window":
            self._trace = None
            self._status.setText("Reconstruction needs a meeting01 window.")
            self._layout_widget.clear()
            return
        try:
            self._trace = self.repo.adapter(adapter_key).load_latent(node)
        except NotImplementedError:
            self._trace = None
            self._status.setText("Reconstruction needs a meeting01 window.")
            return
        except (BindingUnavailableError, RuntimeError) as exc:
            self._trace = None
            self._status.setText(str(exc))
            self._layout_widget.clear()
            self._metrics.setRowCount(0)
            return
        self._render()

    def can_export(self) -> bool:
        return self._trace is not None

    def export_figure(self, path, **opts):
        from experiment_microscope.viz.mpl_export import annotate_provenance, new_figure, save_figure

        t = self._trace
        orig = np.asarray(t.original).reshape(-1)
        rec = np.asarray(t.reconstruction).reshape(-1)
        n = min(orig.size, rec.size)
        fig = new_figure(width_in=opts.get("width_in", 6.5), height_in=opts.get("height_in", 4.0))
        ax1 = fig.add_subplot(211)
        ax1.plot(orig[:n], lw=0.8, label="original")
        ax1.plot(rec[:n], lw=0.8, label="reconstruction")
        ax1.legend(fontsize=6)
        ax1.set_title("SNN autoencoder reconstruction")
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(orig[:n] - rec[:n], lw=0.7, color="0.4")
        ax2.set_title("residual")
        ax2.set_xlabel("flattened index")
        mse = t.metrics.get("mse")
        annotate_provenance(ax1, f"origin: {t.origin.value}; mse={mse.display() if mse else '—'}")
        return save_figure(fig, path, transparent=opts.get("transparent", False))

    def _render(self) -> None:
        t = self._trace
        orig = np.asarray(t.original).reshape(-1)
        rec = np.asarray(t.reconstruction).reshape(-1)
        n = min(orig.size, rec.size)
        orig, rec = orig[:n], rec[:n]
        x = np.arange(n)
        self._layout_widget.clear()

        p0 = self._layout_widget.addPlot(row=0, col=0, title="original vs reconstruction")
        p0.addLegend()
        p0.plot(x, orig, pen=pg.mkPen((120, 170, 255)), name="original")
        p0.plot(x, rec, pen=pg.mkPen((255, 170, 90)), name="reconstruction")
        p0.showGrid(x=True, y=True, alpha=0.2)
        p1 = self._layout_widget.addPlot(row=1, col=0, title="residual (original − reconstruction)")
        p1.setXLink(p0)
        p1.plot(x, orig - rec, pen=pg.mkPen((150, 150, 150)))
        p1.showGrid(x=True, y=True, alpha=0.2)

        self._metrics.setRowCount(len(t.metrics) + 2)
        r = 0
        for name, val in t.metrics.items():
            self._metrics.setItem(r, 0, QTableWidgetItem(name))
            self._metrics.setItem(r, 1, QTableWidgetItem(val.labelled() if val else "—"))
            r += 1
        for name, v in (("latent_dim", np.asarray(t.latent).size), ("length", n)):
            self._metrics.setItem(r, 0, QTableWidgetItem(name))
            self._metrics.setItem(r, 1, QTableWidgetItem(str(v)))
            r += 1
        self._status.setText(f"reconstruction  [{t.origin.value}] — {n} points, latent dim "
                             f"{np.asarray(t.latent).size}")
