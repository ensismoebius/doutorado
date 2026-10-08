"""Renders `live.demo.LiveVowelSnnNodesDemo`'s snapshot: the SNN itself as
a network diagram -- input, hidden and output neurons as nodes, lit by
their own recent activity, joined by the actual learned weights (`w1`,
`w2`) -- instead of `LiveSpikeView`'s time-over-neurons raster. Both
widgets read the exact same `live.demo.LiveVowelSnnDemoBase.snapshot()`
dict and share the confidence bar chart (`widgets/_live_readouts.py`);
this one is purely a different picture of the same live state, not a
second pipeline.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, NEUTRAL_COLOR, SNN_COLOR, TEXT_COLOR
from efficient_nn_lab.widgets._live_readouts import render_confidence_bars
from efficient_nn_lab.widgets._mpl_perf import fast_clear

_INPUT_X, _HIDDEN_X, _OUTPUT_X = 0.12, 0.5, 0.88
#: A silent node or a zero-weight edge still gets drawn, just barely --
#: fully invisible would read as "this neuron/connection doesn't exist"
#: instead of "it exists and happens to be quiet right now".
_MIN_ALPHA = 0.12


class LiveNodeView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 3.6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax_top, self._ax_bottom = self._figure.subplots(2, 1, height_ratios=[3, 1])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, snapshot: dict[str, object]) -> None:
        fast_clear(self._ax_top)
        fast_clear(self._ax_bottom)
        self._ax_top.set_title("")
        self._ax_top.set_aspect("auto")
        self._ax_bottom.set_title("")
        self._ax_bottom.set_aspect("auto")

        if not snapshot.get("has_weights", False):
            self._ax_top.text(
                0.5, 0.5, "Sem pesos treinados ainda -- rode live.train antes do demo",
                ha="center", va="center", transform=self._ax_top.transAxes, color=NEUTRAL_COLOR,
            )
            self._ax_top.set_xticks([])
            self._ax_top.set_yticks([])
            self._ax_bottom.set_xticks([])
            self._ax_bottom.set_yticks([])
            self._canvas.draw_idle()
            return

        self._render_network(snapshot)
        render_confidence_bars(self._ax_bottom, snapshot)
        self._canvas.draw_idle()

    def _render_network(self, snapshot: dict[str, object]) -> None:
        ax = self._ax_top
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([])
        ax.set_yticks([])

        w1 = np.asarray(snapshot["w1"])
        w2 = np.asarray(snapshot["w2"])
        n_in, n_hidden = w1.shape
        n_out = w2.shape[1]
        intensity = np.asarray(snapshot["input_intensity"])
        hidden_rate = np.asarray(snapshot["hidden_rate"])
        confidences = np.asarray(snapshot["class_confidences"])
        names = list(snapshot["class_names"])
        predicted = snapshot.get("predicted_class")

        in_y = np.linspace(0.05, 0.95, n_in) if n_in > 1 else np.array([0.5])
        hid_y = np.linspace(0.05, 0.95, n_hidden) if n_hidden > 1 else np.array([0.5])
        out_y = np.linspace(0.15, 0.85, n_out) if n_out > 1 else np.array([0.5])

        self._draw_edges(ax, w1, _INPUT_X, in_y, _HIDDEN_X, hid_y)
        self._draw_edges(ax, w2, _HIDDEN_X, hid_y, _OUTPUT_X, out_y)

        in_colors = [to_rgba(NEUTRAL_COLOR, a) for a in np.clip(intensity, _MIN_ALPHA, 1.0)]
        ax.scatter(np.full(n_in, _INPUT_X), in_y, s=26, c=in_colors, zorder=5, linewidths=0)

        hid_colors = [to_rgba(SNN_COLOR, a) for a in np.clip(hidden_rate, _MIN_ALPHA, 1.0)]
        ax.scatter(np.full(n_hidden, _HIDDEN_X), hid_y, s=34, c=hid_colors, zorder=5, linewidths=0)

        out_alpha = np.clip(confidences, _MIN_ALPHA, 1.0)
        out_colors = [
            to_rgba(ACCENT_COLOR if names[i] == predicted else BITNET_COLOR, out_alpha[i]) for i in range(n_out)
        ]
        ax.scatter(np.full(n_out, _OUTPUT_X), out_y, s=90, c=out_colors, zorder=5, linewidths=0)
        for i, name in enumerate(names):
            ax.text(
                _OUTPUT_X + 0.05, out_y[i], name.upper(), ha="left", va="center", fontsize=9, color=TEXT_COLOR,
                weight="bold" if name == predicted else "normal",
            )

        ax.text(_INPUT_X, 1.04, f"entrada ({n_in})", ha="center", va="bottom", fontsize=8, color=NEUTRAL_COLOR)
        ax.text(_HIDDEN_X, 1.04, f"oculta ({n_hidden})", ha="center", va="bottom", fontsize=8, color=SNN_COLOR)
        ax.text(_OUTPUT_X, 1.04, "saída", ha="center", va="bottom", fontsize=8, color=BITNET_COLOR)
        status = "capturando" if snapshot.get("is_capturing") else "parado"
        ax.set_title(f"Neurônios da SNN e os pesos aprendidos -- {status}", fontsize=9.5)

    def _draw_edges(
        self, ax, weights: np.ndarray, x0: float, y0_arr: np.ndarray, x1: float, y1_arr: np.ndarray
    ) -> None:
        """One `LineCollection` (a single artist) per layer pair, not one
        `Line2D` per connection -- at up to a few hundred edges per layer,
        that many separate artists would itself be the frame-budget cost
        `widgets/_mpl_perf.py` exists to avoid."""
        max_abs = float(np.abs(weights).max())
        if max_abs <= 1e-12:
            return
        norm = np.abs(weights) / max_abs
        n0, n1 = weights.shape
        segments = [[(x0, y0_arr[i]), (x1, y1_arr[j])] for i in range(n0) for j in range(n1)]
        colors = [
            to_rgba(BITNET_COLOR if weights[i, j] >= 0 else SNN_COLOR, _MIN_ALPHA + 0.5 * norm[i, j])
            for i in range(n0)
            for j in range(n1)
        ]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.6, zorder=1))
