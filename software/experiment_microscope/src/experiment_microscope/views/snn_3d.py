"""3D SNN activity view (FIXME §18).

Renders the meeting01 SNN-AE encoder as columns of neurons:

    encoded input  →  Linear(64)  →  LIF spikes(64)  →  latent(32)

* node position  = (layer index, neuron index)
* node size / colour = activity for the selected window (|activation|; spike
  count for the LIF layer)
* edge opacity / width = |weight| of the connecting ``Linear`` layer, top-K per
  target neuron only (never every edge of the layer)

``time_steps == 1`` for this experiment, so the LIF trace is a single-step
membrane snapshot, not a trajectory — there is nothing to animate here and the
status line says so. Disabled entirely in low-performance mode (§39).
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.viz.pyvista_panel import PV_OK, PyVistaPanel, pv

_LAYER_GAP = 40.0


class Snn3D(QWidget):
    def __init__(self, repo, app_state=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.app_state = app_state
        self._layers: list[dict] | None = None  # per column: {name, activity, weight_in}
        self._player = None
        self._active_upto = -1  # -1 = all columns lit; k = flood reached column k

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._topk = QSpinBox()
        self._topk.setRange(1, 32)
        self._topk.setValue(3)
        self._wthr = QDoubleSpinBox()
        self._wthr.setRange(0.0, 5.0)
        self._wthr.setSingleStep(0.05)
        self._wthr.setDecimals(2)
        self._wthr.setValue(0.0)
        self._athr = QDoubleSpinBox()
        self._athr.setRange(0.0, 1.0)
        self._athr.setSingleStep(0.02)
        self._athr.setDecimals(2)
        self._athr.setValue(0.0)
        for w in (self._topk, self._wthr, self._athr):
            w.valueChanged.connect(self._render)
        for lbl, w in (
            ("top-K edges / neuron", self._topk),
            ("|weight| threshold", self._wthr),
            ("activity threshold", self._athr),
        ):
            bar.addWidget(QLabel(lbl))
            bar.addWidget(w)
        self._status = QLabel("Select a meeting01 window with a trained model.")
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        self._panel = PyVistaPanel()
        root.addWidget(self._panel, 1)

    # -- external API -----------------------------------------------------
    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        self._layers = None
        if self.app_state is not None and getattr(self.app_state, "low_performance_mode", False):
            self._status.setText("3D disabled (low-performance mode).")
            return
        if adapter_key != "meeting01" or (getattr(node, "handle", {}) or {}).get("level") != "window":
            self._status.setText("SNN 3D needs an individual meeting01 window.")
            return
        adapter = self.repo.adapter(adapter_key)
        try:
            trace, spec, lif_ok = adapter.load_ae_trace(node)
        except (BindingUnavailableError, NotImplementedError, RuntimeError) as exc:
            self._status.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"load_ae_trace failed: {exc}")
            return

        cols: list[dict] = [
            {"name": "input", "activity": np.abs(np.asarray(trace.encoded_input).reshape(-1)),
             "weight_in": None}
        ]
        for lyr in trace.encoder_layers:
            act = None if lyr.output is None else np.abs(np.asarray(lyr.output).reshape(-1))
            if act is None:
                continue
            name = {"linear": "linear", "lif": "LIF spikes", "other": "layer"}.get(lyr.kind, "layer")
            cols.append({"name": name, "activity": act,
                         "weight_in": None if lyr.weight is None else np.asarray(lyr.weight)})
        self._layers = cols
        self._active_upto = -1
        if self._player is not None:
            # one frame per column: the signal floods input → … → latent
            self._player.set_total_frames(len(cols))
        note = "" if lif_ok else "  — LIF params are constructed defaults (pre-fix checkpoint)"
        self._status.setText(
            f"{spec['architecture']} / {spec['encoding']}  "
            f"({' → '.join(str(c['activity'].size) for c in cols)} neurons); "
            f"edges = top-K |Linear weight|; press ▶ to flood the signal through the layers"
            f"{note}"
        )
        self._render()

    # -- animation (FIXME §17, §24) -----------------------------------
    def set_timeline(self, player) -> None:
        self._player = player
        player.frame_changed.connect(self._on_frame)

    def _on_frame(self, frame: int) -> None:
        if not self._layers:
            return
        self._active_upto = int(frame)
        self._render()

    # -- rendering ------------------------------------------------------
    def _render(self) -> None:
        if not (PV_OK and self._panel.available) or not self._layers:
            return
        p = self._panel.plotter
        p.clear()
        k = self._topk.value()
        wthr = self._wthr.value()
        athr = self._athr.value()

        upto = self._active_upto  # -1 → all columns lit
        centres: list[np.ndarray] = []
        for xi, col in enumerate(self._layers):
            act = col["activity"].astype(float)
            norm = act / (act.max() or 1.0)
            n = act.size
            ys = (np.arange(n) - n / 2.0) * (60.0 / max(n, 1))
            pts = np.c_[np.full(n, xi * _LAYER_GAP), ys, np.zeros(n)]
            centres.append(pts)
            reached = upto < 0 or xi <= upto
            keep = norm >= athr
            if keep.any():
                cloud = pv.PolyData(pts[keep])
                cloud["activity"] = norm[keep] if reached else np.zeros(int(keep.sum()))
                p.add_mesh(cloud, scalars="activity", cmap="inferno",
                           render_points_as_spheres=True, point_size=14 if reached else 7,
                           clim=(0.0, 1.0), opacity=1.0 if reached else 0.25,
                           show_scalar_bar=(xi == 0))

        # edges: Linear weight (out_features, in_features), top-K by |w| per output
        for xi, col in enumerate(self._layers[1:], start=1):
            w = col["weight_in"]
            if w is None:
                continue
            if upto >= 0 and xi > upto:
                continue  # signal has not flooded into this layer yet
            src, dst = centres[xi - 1], centres[xi]
            w = np.asarray(w)
            if w.shape != (dst.shape[0], src.shape[0]):
                continue
            lines = []
            mags = []
            for o in range(w.shape[0]):
                row = np.abs(w[o])
                idx = np.argsort(row)[-k:]
                for i in idx:
                    if row[i] <= wthr:
                        continue
                    lines.append(src[i])
                    lines.append(dst[o])
                    mags.append(row[i])
            if not lines:
                continue
            pa = np.asarray(lines)
            cells = np.hstack([[2, 2 * j, 2 * j + 1] for j in range(len(mags))])
            poly = pv.PolyData(pa, lines=cells)
            poly["|w|"] = np.repeat(mags, 1)
            p.add_mesh(poly, scalars="|w|", cmap="bone", opacity=0.35, line_width=1.5,
                       show_scalar_bar=False)

        for step in (p.reset_camera,):
            try:
                step()
            except Exception:  # noqa: BLE001
                pass
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                p.show_axes()
            except Exception:  # noqa: BLE001
                pass
