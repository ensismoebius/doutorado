"""SNN Lab — encoding → LIF spike response (FIXME §15, §16, §17).

For a meeting01 window this runs the experiment's own stand-alone LIF sweep
(``apply_snn_architecture_transform(encoded, "recurrent", alpha, v_th)``) and
shows, on one time axis:

    normalized window        the z-scored input
    input encoding spikes    direct / poisson / latency spike train
    LIF output spikes        what the recurrent architecture emits

so the researcher can see how the leak (alpha) and threshold (v_th) reshape the
spike pattern. The transform is the C++ implementation — there is no second
Python model here. Per-neuron membrane traces are *not* exposed by the binding
(``snn_ae_forward`` docstring: "added in a later milestone"); this view is
honest about showing spikes only.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.views._timesync import TimeCursor

_ENCODINGS = ("direct", "poisson", "latency")


class SnnLab(QWidget):
    def __init__(
        self,
        repo: DataRepository,
        selection: SelectionState | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self._selection = selection
        self._cursor: TimeCursor | None = None
        self._window: np.ndarray | None = None

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._encoding = QComboBox()
        self._encoding.addItems(_ENCODINGS)
        self._encoding.setCurrentText("latency")
        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.10, 0.99)
        self._alpha.setSingleStep(0.05)
        self._alpha.setValue(0.90)
        self._alpha.setPrefix("α ")
        self._vth = QDoubleSpinBox()
        self._vth.setRange(0.10, 3.0)
        self._vth.setSingleStep(0.10)
        self._vth.setValue(1.0)
        self._vth.setPrefix("v_th ")
        self._seed = QSpinBox()
        self._seed.setRange(0, 2**31 - 1)
        for w in (self._encoding, self._alpha, self._vth, self._seed):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(self._render)
        for lbl, w in (("encoding", self._encoding), ("", self._alpha), ("", self._vth), ("seed", self._seed)):
            if lbl:
                bar.addWidget(QLabel(lbl))
            bar.addWidget(w)
        self._status = QLabel("Select a meeting01 window.")
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        if not PG_OK:
            root.addWidget(missing_widget("SNN Lab"))
            self._layout_widget = None
            return
        self._layout_widget = pg.GraphicsLayoutWidget()
        root.addWidget(self._layout_widget, 1)

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._layout_widget is None:
            return
        h = getattr(node, "handle", {}) or {}
        if adapter_key != "meeting01" or h.get("level") != "window":
            self._window = None
            self._status.setText("SNN Lab needs a meeting01 window.")
            self._layout_widget.clear()
            return
        try:
            sig: Signal1D = self.repo.adapter(adapter_key).load_signal(node)
        except BindingUnavailableError as exc:
            self._window = None
            self._status.setText(str(exc))
            return
        self._window = np.asarray(sig.samples, dtype=float).reshape(-1)
        self._label = sig.label
        self._render()

    def _render(self) -> None:
        if self._layout_widget is None or self._window is None:
            return
        from experiment_microscope.processing import meeting01 as m

        enc_name = self._encoding.currentText()
        try:
            col = self._window.reshape(-1, 1)
            encoded = np.asarray(m.encode(col, enc_name, self._seed.value()))
            lif = np.asarray(m.architecture_transform(
                encoded, "recurrent", self._alpha.value(), self._vth.value()))
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"transform failed: {exc}")
            return

        enc1 = encoded.reshape(-1)
        lif1 = lif.reshape(-1)
        t = np.arange(enc1.size)
        in_spikes = np.flatnonzero(enc1 > 0.5)
        out_spikes = np.flatnonzero(lif1 > 0.5)
        self._layout_widget.clear()

        p0 = self._layout_widget.addPlot(row=0, col=0, title="normalized window (z-score)")
        p0.plot(np.arange(self._window.size), self._window, pen=pg.mkPen((120, 170, 255)))
        p0.showGrid(x=True, y=True, alpha=0.2)
        if self._selection is not None:
            if self._cursor is None:
                self._cursor = TimeCursor(p0, self._selection)
            else:
                self._cursor.rebind(p0)

        p1 = self._layout_widget.addPlot(row=1, col=0, title=f"{enc_name} encoding — {in_spikes.size} spike(s)")
        p1.setXLink(p0)
        _raster(p1, in_spikes, (255, 190, 90))

        p2 = self._layout_widget.addPlot(
            row=2, col=0,
            title=f"recurrent LIF output — {out_spikes.size} spike(s) "
                  f"(α={self._alpha.value():.2f}, v_th={self._vth.value():.2f})")
        p2.setXLink(p0)
        _raster(p2, out_spikes, (90, 220, 140))

        kept = np.intersect1d(in_spikes, out_spikes).size
        self._status.setText(
            f"{getattr(self, '_label', 'window')}  [computed] — "
            f"in {in_spikes.size} → out {out_spikes.size} spikes "
            f"({kept} coincident, {out_spikes.size - kept} LIF-added, "
            f"{in_spikes.size - kept} LIF-suppressed); per-neuron v_mem not exposed by the binding"
        )


def _raster(plot, spike_idx: np.ndarray, color) -> None:
    plot.setYRange(0.0, 1.2)
    plot.showGrid(x=True, alpha=0.2)
    if spike_idx.size:
        plot.plot(spike_idx, np.ones_like(spike_idx, dtype=float), pen=None,
                  symbol="|", symbolSize=14, symbolPen=pg.mkPen(color, width=2))
