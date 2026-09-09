"""3D wavelet coefficient landscape (FIXME §11).

    X = coefficient index within a packet leaf
    Y = packet leaf (low → high frequency)
    Z = coefficient magnitude

The same packet decomposition the 2D Wavelet Lab shows, lifted into a surface so
a whole sub-band structure is visible at once. Rotation / zoom / pan come from
VTK; a threshold slider clips low-magnitude coefficients; a spin box lifts one
leaf out for inspection. Disabled entirely in low-performance mode (§39).
"""

from __future__ import annotations

from experiment_microscope.views._help import HelpBox

import os

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
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

_WAVELETS = ["haar", "daub4", "daub6", "daub8", "daub10", "daub12", "daub20"]


_HELP = """
<b>What this shows.</b> The wavelet-packet decomposition as a 3-D surface instead
of a table.
<br><br>
<b>X</b> = coefficient index within a leaf. <b>Y</b> = leaf number, low to high
frequency. <b>Z</b> (height & colour) = |coefficient| magnitude, normalised so
the largest is 1. Ridges are bands carrying a lot of the signal's power.
<br><br>
The <b>|z| threshold</b> slider hides small coefficients; <b>isolate leaf</b>
lifts one band out as a red line. Rotate / zoom / pan with the mouse. Disabled in
low-performance mode.
"""


class Wavelet3D(QWidget):
    def __init__(self, repo, app_state=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.app_state = app_state
        self._signal = None
        self._grid_z: np.ndarray | None = None

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Wavelet 3D', _HELP))
        bar = QHBoxLayout()
        self._wavelet = QComboBox()
        self._wavelet.addItems(_WAVELETS)
        self._level = QSpinBox()
        self._level.setRange(1, 7)
        self._level.setValue(4)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setSingleStep(0.02)
        self._threshold.setDecimals(2)
        self._threshold.setValue(0.0)
        self._leaf = QSpinBox()
        self._leaf.setRange(-1, 0)
        self._leaf.setSpecialValueText("all")
        self._leaf.setValue(-1)
        for w in (self._wavelet, self._level):
            w.currentIndexChanged.connect(self._recompute) if isinstance(w, QComboBox) \
                else w.valueChanged.connect(self._recompute)
        self._threshold.valueChanged.connect(self._render)
        self._leaf.valueChanged.connect(self._render)
        for lbl, w in (("wavelet", self._wavelet), ("level", self._level),
                       ("|z| threshold", self._threshold), ("isolate leaf", self._leaf)):
            bar.addWidget(QLabel(lbl))
            bar.addWidget(w)
        self._status = QLabel("Select a signal.")
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        self._panel = PyVistaPanel()
        root.addWidget(self._panel, 1)

    # -- external API ------------------------------------------
    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self.app_state is not None and self.app_state.low_performance_mode:
            self._status.setText("3D disabled (low-performance mode).")
            return
        try:
            self._signal = self.repo.adapter(adapter_key).load_signal(node)
        except NotImplementedError:
            self._signal = None
            self._status.setText("Selected object has no 1-D signal.")
            return
        except BindingUnavailableError as exc:
            self._signal = None
            self._status.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._signal = None
            self._status.setText(f"load_signal failed: {exc}")
            return
        self._recompute()

    # -- internals --------------------------------------------
    def _recompute(self) -> None:
        if self._signal is None:
            return
        from experiment_microscope.processing import wavelet as wl

        raw = np.asarray(self._signal.samples, dtype=float)
        data = raw[0] if raw.ndim == 2 else raw.ravel()
        try:
            d = wl.decompose(data, self._wavelet.currentText(), "packet", self._level.value())
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"decompose failed: {exc}")
            return
        if not d.leaf_count:
            self._status.setText("packet decomposition produced no leaves.")
            self._grid_z = None
            return
        per = d.transformed_signal.size // d.leaf_count
        z = np.abs(d.transformed_signal[: per * d.leaf_count].reshape(d.leaf_count, per))
        mx = float(z.max()) or 1.0
        self._grid_z = z / mx  # normalized magnitude, rows = leaves
        self._leaf.blockSignals(True)
        self._leaf.setRange(-1, d.leaf_count - 1)
        self._leaf.blockSignals(False)
        self._status.setText(
            f"{self._signal.label or 'signal'} — {d.leaf_count} leaves x {per} coeffs "
            f"[computed], magnitude normalized to max"
        )
        self._render()

    def _render(self) -> None:
        if not (PV_OK and self._panel.available) or self._grid_z is None:
            return
        z = self._grid_z.copy()
        t = self._threshold.value()
        if t > 0:
            z[z < t] = np.nan
        iso = self._leaf.value()
        n_leaves, per = z.shape
        xs = np.arange(per)
        ys = np.arange(n_leaves)
        xx, yy = np.meshgrid(xs, ys)
        grid = pv.StructuredGrid(xx.astype(float), yy.astype(float) * 4.0, z * 20.0)
        grid["|z| / max"] = z.ravel(order="C")
        p = self._panel.plotter
        p.clear()
        p.add_mesh(grid, scalars="|z| / max", cmap="viridis", nan_opacity=0.0,
                   show_scalar_bar=True, lighting=True)
        if 0 <= iso < n_leaves:
            row = z[iso].copy()
            rx = np.arange(per, dtype=float)
            ry = np.full(per, iso * 4.0)
            line = pv.lines_from_points(np.c_[rx, ry, np.nan_to_num(row) * 20.0 + 0.5])
            p.add_mesh(line, color="red", line_width=3)
        steps = [lambda: p.set_scale(zscale=1.0), p.reset_camera]
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            steps.insert(1, p.show_axes)  # needs a live interactor
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 - headless VTK has no interactor
                pass
