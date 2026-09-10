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
from experiment_microscope.views._help import HelpBox, fill_metric_table, metric_table
from experiment_microscope.views._pg import PG_OK, missing_widget, pg

_HELP = """
<b>What this shows.</b> One audio <b>window</b> (a 256-sample slice) after it has
been pushed through a trained <b>SNN autoencoder</b> (SNN = Spiking Neural
Network; an autoencoder squeezes the window into a small <b>latent</b> vector of
32 numbers and then rebuilds it).
<br><br>
<b>Top plot.</b> Blue = the <i>original</i> encoder input (the flattened,
spike-encoded window). Orange = the <i>reconstruction</i> the decoder produced
from the latent vector. The closer they sit, the more information the 32-number
latent kept.
<br><b>Bottom plot.</b> The <i>residual</i> = original − reconstruction, point by
point. A flat line near zero means a faithful rebuild; spikes mark where the
model lost detail.
<br><br>
<b>The numbers</b> (table below): every one has a "what it means" column.
MSE / MAE measure the error size (0 = perfect); R² is the fraction of the
signal's variability captured (1 = perfect, 0 = no better than a flat line);
Pearson r is shape agreement ignoring scale (+1 = identical shape).
<br><br>
<b>Origin tag</b> <code>[computed]</code> means these curves were recomputed here
through the exact experiment code, not read from a cached file.
"""


class ReconstructionView(QWidget):
    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._trace = None

        root = QVBoxLayout(self)
        root.addWidget(HelpBox("Reconstruction", _HELP))
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
        self._metrics = metric_table()
        split.addWidget(self._metrics)
        split.setSizes([440, 170])
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

        from experiment_microscope.views._help import label_plot

        from experiment_microscope.core import palette, verdict

        r2m = t.metrics.get("r2")
        headline = verdict.reconstruction(
            None if r2m is None or r2m.is_missing else float(r2m.magnitude))
        p0 = self._layout_widget.addPlot(row=0, col=0)
        label_plot(p0, bottom="index within the flattened window (sample number)",
                   left="amplitude (z-scored, unitless)", title=headline)
        p0.plot(x, orig, pen=palette.pen("input", 2), name="original (what went in)")
        p0.plot(x, rec, pen=palette.pen("output", 2), name="rebuild (what came out)")
        resid = orig - rec
        j = int(np.argmax(np.abs(resid))) if resid.size else 0
        p1 = self._layout_widget.addPlot(row=1, col=0)
        label_plot(p1, bottom="index within the flattened window (sample number)",
                   left="original − rebuild",
                   title="What was lost — the flatter this line, the better the rebuild",
                   legend=False)
        p1.setXLink(p0)
        p1.plot(x, resid, pen=palette.pen("error"), name="difference")
        if resid.size:
            p1.plot([x[j]], [resid[j]], pen=None, symbol="o", symbolSize=10,
                    symbolBrush=palette.brush("highlight"), name="biggest miss")
            txt = pg.TextItem("biggest miss here", color=palette.rgb("highlight"), anchor=(0.5, 1.2))
            txt.setPos(float(x[j]), float(resid[j]))
            p1.addItem(txt)

        from experiment_microscope.views._plotinfo import HoverReadout, set_source

        src = f"meeting01.snn_ae_forward() · origin [{t.origin.value}]"
        set_source(p0, src)
        set_source(p1, src)
        self._hovers = [HoverReadout(p0, x_label="sample"),
                        HoverReadout(p1, x_label="sample")]

        _METRIC_NAMES = {
            "mse": "MSE", "mae": "MAE", "r2": "R2", "pearson_r": "Pearson r",
            "lif_params": "LIF parameters",
        }
        rows: list[tuple[str, str, str]] = []
        for name, val in t.metrics.items():
            rows.append((_METRIC_NAMES.get(name, name),
                         (val.labelled() if val else "—"), ""))
        rows.append(("latent dimension", str(np.asarray(t.latent).size),
                     "how many numbers the encoder compressed the window into"))
        rows.append(("length", str(n),
                     "number of points in the window / reconstruction"))
        fill_metric_table(self._metrics, rows)
        self._status.setText(
            f"Reconstruction of one window  ·  origin [{t.origin.value}] (recomputed "
            f"through the experiment code)  ·  {n} points  ·  latent = "
            f"{np.asarray(t.latent).size} numbers")
