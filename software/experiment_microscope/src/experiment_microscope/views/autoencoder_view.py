"""The whole autoencoder as one graph — encoder + decoder together (FIXME §18, §19).

    input(256) → Linear(64) → LIF(64) → latent(32) → Linear(64) → LIF(64) → recon(256)

Drawn left→right as columns of neurons joined by their ``Linear`` weights.

* node colour / size  = that neuron's activation for the selected window
* edge colour         = weight sign (blue ``+`` / red ``−``), only the top few
                        ``|weight|`` per target neuron so the picture stays legible
* press ▶ on the transport bar → the signal **floods** column by column, in
  through the encoder to the latent code and back out through the decoder
* **zoom in** (mouse wheel) on a spiking column, or tick *Neuron detail* → each
  visible LIF neuron shows its membrane charge, its firing line and whether it
  fired; click any neuron for its exact numbers

Everything is recomputed through ``nn_microscope`` — there is no second
implementation of the network here.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from experiment_microscope.core.i18n import t as _t
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._help import HelpBox

try:  # pyqtgraph is a hard dep of the app, but keep the view importable without it
    import pyqtgraph as pg

    _PG = True
except Exception:  # noqa: BLE001  pragma: no cover
    _PG = False

_MAX_NODES = 44          # per column; bigger layers are evenly sub-sampled
_TOPK = 3                # strongest incoming edges drawn per target neuron
_GAP = 1.0               # x distance between columns
_DETAIL_NODES = 18       # switch a column to per-neuron gauges below this many visible

_HELP = """
<b>What this panel shows.</b> The whole trained autoencoder — the half that
<i>compresses</i> (encoder) and the half that <i>rebuilds</i> (decoder) — as one
left-to-right drawing: input (256) → 64 → <b>latent code (32)</b> → 64 →
reconstruction (256).
<br><br>
<b>Every dot is a neuron.</b> Colour and size show how strongly it lit up for the
chosen window. <b>The lines are the weights</b>: blue adds, red subtracts; only the
strongest few per neuron are drawn or it becomes a smear.
<br><br>
<b>Press ▶</b> on the transport bar to watch the signal <b>travel through</b> the
network one column at a time, in to the latent code and back out.
<br><br>
<b>Zoom in</b> (mouse wheel) on a spiking column, or tick <i>Neuron detail</i>:
each LIF neuron then shows its built-up charge, its firing line and whether it
fired. Click a neuron for its exact numbers. LIF = leaky integrate-and-fire neuron.
"""


def _lerp_brush(v: float):
    """0 → dark slate, 1 → bright amber. v already in [0, 1]."""
    v = 0.0 if not np.isfinite(v) else float(min(max(v, 0.0), 1.0))
    lo = np.array([40, 44, 60])
    hi = np.array([255, 196, 64])
    r, g, b = (lo + (hi - lo) * v).astype(int)
    return pg.mkBrush(r, g, b, 235)


class AutoencoderView(QWidget):
    def __init__(self, repo, selection=None, app_state=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.selection = selection
        self.app_state = app_state
        self._cols: list[dict] | None = None       # per column: name/act/weight_in/kind/vmem/vth/half
        self._sample: list[np.ndarray] = []        # per column: original indices kept
        self._player = None
        self._reveal = -1                          # -1 = whole net lit; k = flood reached column k
        self._detail = False
        self._rendering = False

        root = QVBoxLayout(self)
        root.addWidget(HelpBox("Autoencoder graph", _HELP))

        bar = QHBoxLayout()
        self._detail_cb = QCheckBox(_t("Neuron detail"))
        self._detail_cb.toggled.connect(self._on_detail_toggled)
        bar.addWidget(self._detail_cb)
        self._readout = QLabel(_t("Select a meeting01 window with a trained model."))
        self._readout.setWordWrap(True)
        bar.addWidget(self._readout, 1)
        root.addLayout(bar)

        if _PG:
            self._plot = pg.PlotWidget()
            self._plot.setBackground(None)
            self._plot.setMenuEnabled(False)
            self._plot.showGrid(x=False, y=False)
            self._plot.getPlotItem().hideAxis("left")
            self._plot.getPlotItem().hideAxis("bottom")
            self._plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)
            root.addWidget(self._plot, 1)
        else:  # pragma: no cover
            self._plot = None
            root.addWidget(QLabel("pyqtgraph unavailable — cannot draw the graph."))

    # -- external API --------------------------------------------------
    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        self._cols = None
        if self._plot is None:
            return
        h = getattr(node, "handle", {}) or {}
        if adapter_key != "meeting01" or h.get("level") != "window":
            self._readout.setText(_t("The autoencoder graph needs an individual meeting01 window."))
            self._plot.clear()
            return
        adapter = self.repo.adapter(adapter_key)
        try:
            trace, spec, lif_ok = adapter.load_ae_trace(node)
        except (BindingUnavailableError, NotImplementedError, RuntimeError) as exc:
            self._readout.setText(str(exc))
            self._plot.clear()
            return
        except Exception as exc:  # noqa: BLE001
            self._readout.setText(_t("load_ae_trace failed: {exc}", exc=exc))
            self._plot.clear()
            return

        cols: list[dict] = [{
            "name": _t("input"), "half": "in", "kind": "input",
            "act": np.abs(np.asarray(trace.encoded_input).reshape(-1)),
            "weight_in": None, "vmem": None, "vth": None,
        }]
        n_enc = len(trace.encoder_layers)
        for j, lyr in enumerate((*trace.encoder_layers, *trace.decoder_layers)):
            if lyr.output is None:
                continue
            half = "enc" if j < n_enc else "dec"
            act = np.abs(np.asarray(lyr.output).reshape(-1))
            cols.append({
                "name": lyr.kind, "half": half, "kind": lyr.kind, "act": act,
                "weight_in": None if lyr.weight is None else np.asarray(lyr.weight),
                "vmem": None if lyr.v_mem is None else np.asarray(lyr.v_mem).reshape(-1),
                "vth": lyr.voltage_threshold,
            })
        # the last encoder column IS the latent code; the last column IS the reconstruction
        if 1 <= n_enc < len(cols):
            cols[n_enc]["name"] = _t("latent")
            cols[n_enc]["kind"] = "latent"
        cols[-1]["name"] = _t("reconstruction")
        cols[-1]["kind"] = "recon"

        self._cols = cols
        self._sample = [self._pick(c["act"].size) for c in cols]
        self._reveal = -1
        if self._player is not None:
            self._player.set_total_frames(len(cols))
        note = "" if lif_ok else _t("  — LIF params are constructed defaults (pre-fix checkpoint)")
        self._readout.setText(_t(
            "{arch} / {enc} · {shape} neurons — press ▶ to send the signal through{note}",
            arch=spec["architecture"], enc=spec["encoding"],
            shape=" → ".join(str(c["act"].size) for c in cols), note=note,
        ))
        self._render()

    def set_timeline(self, player) -> None:
        self._player = player
        player.frame_changed.connect(self._on_frame)

    # -- interaction --------------------------------------------------
    def _on_frame(self, frame: int) -> None:
        if not self._cols:
            return
        self._reveal = int(frame)
        self._render()

    def _on_detail_toggled(self, on: bool) -> None:
        self._detail = bool(on)
        self._render()

    def _on_range_changed(self, *_a) -> None:
        # auto-enter detail mode on a deep zoom so a presenter need not know the trick
        if self._rendering or not self._cols or self._detail_cb.isChecked():
            return
        want = self._deep_zoom()
        if want != self._detail:
            self._detail = want
            self._render()

    def _deep_zoom(self) -> bool:
        if self._plot is None or not self._cols:
            return False
        (x0, x1), (y0, y1) = self._plot.getViewBox().viewRange()
        if (x1 - x0) > 2.2 * _GAP:
            return False
        # a column is "detailed" when few of its nodes fall inside the y window
        span = max(y1 - y0, 1e-6)
        return span < 0.55

    # -- rendering --------------------------------------------------
    def _pick(self, n: int) -> np.ndarray:
        if n <= _MAX_NODES:
            return np.arange(n)
        return np.linspace(0, n - 1, _MAX_NODES).round().astype(int)

    def _ys(self, kept: np.ndarray, n: int) -> np.ndarray:
        if n <= 1:
            return np.zeros(len(kept))
        return kept / (n - 1)  # normalised column height in [0, 1]

    def _render(self) -> None:
        if self._plot is None or not self._cols or self._rendering:
            return
        self._rendering = True
        try:
            self._render_body()
        finally:
            self._rendering = False

    def _render_body(self) -> None:
        self._plot.clear()
        vb = self._plot.getViewBox()
        reveal = self._reveal
        centres: list[np.ndarray] = []

        # edges first so nodes sit on top
        for xi, col in enumerate(self._cols):
            kept = self._sample[xi]
            centres.append(np.c_[np.full(len(kept), xi * _GAP), self._ys(kept, col["act"].size)])
        for xi in range(1, len(self._cols)):
            if reveal >= 0 and xi > reveal:
                continue
            w = self._cols[xi]["weight_in"]
            src_all, dst_all = self._sample[xi - 1], self._sample[xi]
            src_c, dst_c = centres[xi - 1], centres[xi]
            pos_x, pos_y, neg_x, neg_y = [], [], [], []
            if w is not None and w.ndim == 2:
                src_pos = {v: k for k, v in enumerate(src_all)}
                for di, o in enumerate(dst_all):
                    row = w[o] if o < w.shape[0] else None
                    if row is None:
                        continue
                    order = np.argsort(np.abs(row))[-_TOPK:]
                    for i in order:
                        si = src_pos.get(int(i))
                        if si is None:
                            continue
                        (pos_x if row[i] >= 0 else neg_x).extend([src_c[si][0], dst_c[di][0]])
                        (pos_y if row[i] >= 0 else neg_y).extend([src_c[si][1], dst_c[di][1]])
            else:  # LIF / identity pass-through: 1:1 links between shared indices
                dst_pos = {v: k for k, v in enumerate(dst_all)}
                for si, s in enumerate(src_all):
                    di = dst_pos.get(int(s))
                    if di is None:
                        continue
                    pos_x.extend([src_c[si][0], dst_c[di][0]])
                    pos_y.extend([src_c[si][1], dst_c[di][1]])
            if pos_x:
                self._plot.addItem(pg.PlotCurveItem(
                    np.asarray(pos_x), np.asarray(pos_y), connect="pairs",
                    pen=pg.mkPen(70, 130, 230, 90)))
            if neg_x:
                self._plot.addItem(pg.PlotCurveItem(
                    np.asarray(neg_x), np.asarray(neg_y), connect="pairs",
                    pen=pg.mkPen(230, 90, 90, 90)))

        # nodes
        for xi, col in enumerate(self._cols):
            kept = self._sample[xi]
            act = col["act"][kept].astype(float)
            norm = act / (col["act"].max() or 1.0)
            reached = reveal < 0 or xi <= reveal
            highlight = xi == reveal
            spots = []
            for k, idx in enumerate(kept):
                v = float(norm[k]) if reached else 0.0
                spots.append({
                    "pos": (centres[xi][k][0], centres[xi][k][1]),
                    "size": (16 if highlight else 12) if reached else 7,
                    "brush": _lerp_brush(v) if reached else pg.mkBrush(60, 60, 70, 120),
                    "pen": pg.mkPen("w", width=1) if highlight else pg.mkPen(None),
                    "data": (xi, int(idx)),
                })
            sp = pg.ScatterPlotItem(spots=spots)
            sp.sigClicked.connect(self._on_click)
            self._plot.addItem(sp)
            lbl = pg.TextItem(col["name"], anchor=(0.5, 1.0), color=(150, 150, 160))
            lbl.setPos(xi * _GAP, 1.06)
            self._plot.addItem(lbl)

        if self._detail:
            self._render_detail(centres)

        vb.setLimits(xMin=-0.6, xMax=(len(self._cols) - 1) * _GAP + 0.6, yMin=-0.15, yMax=1.25)
        if reveal < 0 and not self._detail:
            vb.autoRange(padding=0.08)

    def _render_detail(self, centres: list[np.ndarray]) -> None:
        """Per-neuron gauges for the LIF columns currently in view."""
        (x0, x1), (y0, y1) = self._plot.getViewBox().viewRange()
        for xi, col in enumerate(self._cols):
            if col["vmem"] is None or col["vth"] is None:
                continue
            if xi * _GAP < x0 - 0.3 or xi * _GAP > x1 + 0.3:
                continue
            kept = self._sample[xi]
            vth = float(col["vth"])
            span = max(np.abs(col["vmem"]).max(), vth, 1e-6)
            visible = [(k, i) for k, i in enumerate(kept)
                       if y0 <= centres[xi][k][1] <= y1]
            if not visible or len(visible) > _DETAIL_NODES:
                continue
            x = xi * _GAP
            gw = 0.16
            for k, i in visible:
                yc = centres[xi][k][1]
                vm = float(col["vmem"][i])
                fired = vm >= vth
                # charge bar (0 → vm), firing line at vth, both scaled into a small band
                h = 0.05
                self._plot.addItem(pg.PlotCurveItem(
                    [x - gw, x - gw], [yc - h, yc - h + 2 * h * (vm / span)],
                    pen=pg.mkPen(255, 170, 60, 255, width=3)))
                self._plot.addItem(pg.PlotCurveItem(
                    [x - gw - 0.03, x - gw + 0.03],
                    [yc - h + 2 * h * (vth / span)] * 2,
                    pen=pg.mkPen(230, 90, 90, 220, width=1, style=Qt.PenStyle.DashLine)))
                txt = pg.TextItem(
                    f"n{i}  v={vm:.2f}  vθ={vth:.2f}  {'⚡' if fired else '·'}",
                    anchor=(0, 0.5), color=(220, 200, 120) if fired else (150, 150, 160))
                txt.setPos(x + 0.04, yc)
                self._plot.addItem(txt)

    def _on_click(self, _scatter, points, *_ev) -> None:
        # pyqtgraph ≥0.13: `points` is a numpy array — test with len(), never `not`
        if not self._cols or points is None or len(points) == 0:
            return
        xi, idx = points[0].data()
        col = self._cols[xi]
        half = {"in": "", "enc": _t("Encoder · "), "dec": _t("Decoder · ")}.get(col["half"], "")
        act = float(col["act"][idx])
        msg = _t("{half}{name} — neuron {i} of {n}: activation {a}",
                 half=half, name=col["name"], i=idx, n=col["act"].size, a=f"{act:.3f}")
        if col["vmem"] is not None and col["vth"] is not None:
            vm, vth = float(col["vmem"][idx]), float(col["vth"])
            gap = vm - vth
            verdict = (_t("fired ({g} over the line)", g=f"{gap:+.3f}") if gap >= 0
                       else _t("did not fire ({g} below the line)", g=f"{abs(gap):.3f}"))
            msg += _t("  ·  membrane {vm}, firing line {vth} — {verdict}",
                      vm=f"{vm:.3f}", vth=f"{vth:.3f}", verdict=verdict)
        self._readout.setText(msg)
        if self.selection is not None:
            self.selection.set("neuron", int(idx))
