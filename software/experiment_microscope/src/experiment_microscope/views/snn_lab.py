"""SNN Lab — encoding → LIF spike response (FIXME §15, §16, §17).

For a meeting01 window this runs the experiment's own stand-alone LIF sweep
(``apply_snn_architecture_transform(encoded, "recurrent", alpha, v_th)``) and
shows, on one time axis:

    normalized window        the z-scored input
    input encoding spikes    direct / poisson / latency spike train
    recurrent LIF membrane   v[t] = alpha*v[t-1] + x[t] - s[t-1]*v_th, over the
                             256 window samples (they ARE the time steps here),
                             with the threshold line and spike markers

so the researcher can see how the leak (alpha) and threshold (v_th) reshape both
the membrane trajectory and the spike pattern. The recurrence is the C++
implementation (``recurrent_lif_trace``) — there is no second Python model here.

When a trained SNN-AE ``.npz`` exists for the window's fold, a fourth panel shows
the *autoencoder* encoder LIF layer's per-neuron membrane — that network runs
with ``time_steps == 1`` (the window is fed as a feature vector, not a
sequence), so it is a single-step snapshot, not a trajectory.
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
from experiment_microscope.views._help import HelpBox
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.views._timesync import TimeCursor

_ENCODINGS = ("direct", "poisson", "latency")


_HELP = """
<b>What this shows.</b> How a meeting01 window becomes spikes.
SNN = Spiking Neural Network; its neurons fire discrete 0/1 <b>spikes</b> instead
of sending continuous numbers.
<br><br>
<b>Panel 1</b> — the normalised (z-scored) window.
<b>Panel 2</b> — the input <b>spike train</b> for the chosen encoding
(<i>direct</i> = pass-through, <i>poisson</i> = firing rate ∝ value,
<i>latency</i> = bigger value fires earlier). Each mark is one spike; click it
for its exact time.
<b>Panel 3</b> — the <b>membrane voltage</b> v[t] of the recurrent
Leaky-Integrate-and-Fire (LIF) transform, sweeping the 256 window samples as time
steps: <code>v[t] = α·v[t−1] + x[t] − s[t−1]·v_th</code>. The dashed red line is
the threshold <b>v_th</b>; a green mark sits on every step where v crossed it and
the neuron fired.
<b>Panel 4</b> (only with a trained model) — one membrane value per encoder LIF
neuron, a snapshot because that network runs with a single time step.
<br><br>
<b>Sliders.</b> <b>α</b> (alpha) = leak, 0…1: near 1 the neuron remembers input
longer. <b>v_th</b> = threshold: higher → fewer spikes.
"""



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

        root.addWidget(HelpBox('SNN Lab', _HELP))
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
        from experiment_microscope.core.i18n import t as _t
        self._t = _t
        for lbl, w in (("encoding", self._encoding), ("", self._alpha), ("", self._vth), ("seed", self._seed)):
            if lbl:
                bar.addWidget(QLabel(_t(lbl)))
            bar.addWidget(w)
        self._status = QLabel(_t("Select a meeting01 window."))
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        if not PG_OK:
            root.addWidget(missing_widget("SNN Lab"))
            self._layout_widget = None
            return
        self._layout_widget = pg.GraphicsLayoutWidget()
        root.addWidget(self._layout_widget, 1)
        self._spike_readout = QLabel(_t("click a spike marker for its exact time / membrane / threshold"))
        self._spike_readout.setWordWrap(True)
        root.addWidget(self._spike_readout)
        self._vmem_cache: np.ndarray | None = None

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._layout_widget is None:
            return
        h = getattr(node, "handle", {}) or {}
        if adapter_key != "meeting01" or h.get("level") != "window":
            self._window = None
            self._status.setText(self._t("SNN Lab needs a meeting01 window."))
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
        self._node = node
        self._membrane = None
        try:
            trace, _spec, _lif_ok = self.repo.adapter(adapter_key).load_ae_trace(node)
            for lyr in trace.encoder_layers:
                if lyr.kind == "lif" and lyr.v_mem is not None:
                    self._membrane = (
                        np.asarray(lyr.v_mem).reshape(-1),
                        None if lyr.output is None else np.asarray(lyr.output).reshape(-1),
                        lyr.voltage_threshold,
                    )
                    break
        except Exception:  # noqa: BLE001 - no trained model / still writing
            self._membrane = None
        self._render()

    def _render(self) -> None:
        if self._layout_widget is None or self._window is None:
            return
        from experiment_microscope.processing import meeting01 as m

        enc_name = self._encoding.currentText()
        try:
            col = self._window.reshape(-1, 1)
            encoded = np.asarray(m.encode(col, enc_name, self._seed.value()))
            lif_spk, v_mem = m.recurrent_lif_trace(
                encoded, self._alpha.value(), self._vth.value())
        except Exception as exc:  # noqa: BLE001
            self._status.setText(self._t("transform failed: {exc}", exc=exc))
            return

        enc1 = encoded.reshape(-1)
        lif1 = np.asarray(lif_spk).reshape(-1)
        vmem1 = np.asarray(v_mem).reshape(-1)
        in_spikes = np.flatnonzero(enc1 > 0.5)
        out_spikes = np.flatnonzero(lif1 > 0.5)
        self._vmem_cache = vmem1
        from experiment_microscope.views._plotinfo import HoverReadout, set_source

        self._layout_widget.clear()
        self._hovers = []

        from experiment_microscope.core import palette, verdict
        from experiment_microscope.views._plotinfo import autofit, fade_in

        _t = self._t
        p0 = self._layout_widget.addPlot(row=0, col=0, title=_t("1 · the window going in"))
        p0.plot(np.arange(self._window.size), self._window, pen=palette.pen("input", 2), name="window")
        p0.showGrid(x=True, y=True, alpha=0.2)
        autofit(p0)
        fade_in(self._layout_widget)
        set_source(p0, "nn_microscope.meeting01.recurrent_lif_trace()")
        self._hovers.append(HoverReadout(p0, x_label="step"))
        if self._selection is not None:
            if self._cursor is None:
                self._cursor = TimeCursor(p0, self._selection)
            else:
                self._cursor.rebind(p0)

        p1 = self._layout_widget.addPlot(
            row=1, col=0,
            title=_t("2 · spikes going in — {n} of them (click one to inspect)", n=in_spikes.size))
        p1.setXLink(p0)
        _raster(p1, in_spikes, palette.rgb("spike"), on_click=self._on_input_spike)

        p2 = self._layout_widget.addPlot(
            row=2, col=0,
            title="3 · " + verdict.spikes(in_spikes.size, out_spikes.size, vmem1.size)
                  + _t("  (leak α={a}, firing line v_th={v})",
                       a=f"{self._alpha.value():.2f}", v=f"{self._vth.value():.2f}"))
        p2.setXLink(p0)
        p2.showGrid(x=True, y=True, alpha=0.2)
        p2.setLabel("left", _t("charge inside the neuron  (membrane potential v[t])"))
        p2.setLabel("bottom", _t("time step"))
        p2.plot(np.arange(vmem1.size), vmem1, pen=palette.pen("membrane", 2), name="charge v[t]")
        autofit(p2, x=False)
        self._hovers.append(HoverReadout(p2, x_label="step"))
        p2.addLine(y=float(self._vth.value()),
                   pen=palette.pen("threshold", 1, "dash"))
        if out_spikes.size:
            sc = pg.ScatterPlotItem(
                x=out_spikes, y=vmem1[out_spikes], symbol="t", size=11,
                brush=palette.brush("spike"), pen=pg.mkPen(None), data=list(out_spikes))
            sc.sigClicked.connect(self._on_membrane_spike)
            p2.addItem(sc)

        mem = getattr(self, "_membrane", None)
        if mem is not None:
            v_mem, spk, vth = mem
            neurons = np.arange(v_mem.size)
            n_fire = int((spk > 0.5).sum()) if spk is not None else 0
            p3 = self._layout_widget.addPlot(
                row=3, col=0,
                title=_t("4 · the trained network's 64 encoder neurons — {n} of them "
                         "fired for this window (one charge value each, not a trajectory)", n=n_fire))
            p3.setLabel("bottom", _t("encoder neuron index"))
            p3.setLabel("left", _t("charge at readout"))
            p3.plot(neurons, v_mem, pen=None, symbol="o", symbolSize=6,
                    symbolBrush=palette.brush("membrane"), name="charge")
            self._hovers.append(HoverReadout(p3, x_label="neuron"))
            if vth is not None:
                p3.addLine(y=float(vth), pen=palette.pen("threshold", 1, "dash"))
            if spk is not None and (spk > 0.5).any():
                fi = np.flatnonzero(spk > 0.5)
                p3.plot(fi, v_mem[fi], pen=None, symbol="t", symbolSize=10,
                        symbolBrush=palette.brush("spike"), name="fired")

        kept = np.intersect1d(in_spikes, out_spikes).size
        mem_note = (
            _t("; trained encoder LIF membrane shown ({n} neurons)", n=mem[0].size)
            if getattr(self, "_membrane", None) is not None
            else _t("; no trained .npz for this fold — membrane panel hidden")
        )
        self._status.setText(
            _t("{label}  [computed] — in {i} → out {o} spikes "
               "({k} coincident, {a} LIF-added, {s} LIF-suppressed)",
               label=getattr(self, "_label", "window"), i=in_spikes.size, o=out_spikes.size,
               k=kept, a=out_spikes.size - kept, s=in_spikes.size - kept) + mem_note
        )


    def _on_input_spike(self, _scatter, points) -> None:
        if not points:
            return
        t = int(points[0].data())
        self._spike_readout.setText(
            self._t("input encoding spike — t = sample {t}  ·  encoding = {enc}  ·  seed {seed}  "
                    "(no membrane at the encoder input; it is a fixed spike train)",
                    t=t, enc=self._encoding.currentText(), seed=self._seed.value())
        )
        if self._selection is not None:
            self._selection.set("timestep", t)

    def _on_membrane_spike(self, _scatter, points) -> None:
        if not points or self._vmem_cache is None:
            return
        t = int(points[0].data())
        v = float(self._vmem_cache[t])
        vth = float(self._vth.value())
        self._spike_readout.setText(
            self._t("recurrent-LIF spike — t = sample {t}  ·  v[t] = {v}  ·  "
                    "v_th = {vth}  ·  crossed by {d}  ·  layer = recurrent transform",
                    t=t, v=f"{v:.5f}", vth=f"{vth:.3f}", d=f"{v - vth:+.5f}")
        )
        if self._selection is not None:
            self._selection.set("timestep", t)


def _raster(plot, spike_idx: np.ndarray, color, on_click=None) -> None:
    plot.setYRange(0.0, 1.2)
    plot.showGrid(x=True, alpha=0.2)
    if spike_idx.size:
        sc = pg.ScatterPlotItem(
            x=spike_idx, y=np.ones_like(spike_idx, dtype=float), symbol="|", size=16,
            pen=pg.mkPen(color, width=2), brush=pg.mkBrush(color), data=list(spike_idx))
        if on_click is not None:
            sc.sigClicked.connect(on_click)
        plot.addItem(sc)
