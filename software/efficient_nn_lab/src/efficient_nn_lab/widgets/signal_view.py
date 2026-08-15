"""Renders time-series demos: raw signal + spike raster, or input current +
membrane potential + spike raster. Both spike_generation and lif_dynamics
reveal their trace progressively as the user steps, so the x-axis is fixed
to the full-length window from the first frame instead of auto-rescaling.
A marker at the trace's current tip makes "where are we right now" always
obvious, especially mid-animation.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR


class SignalView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 3.6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax_top, self._ax_bottom = self._figure.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, values: dict[str, object]) -> None:
        self._ax_top.clear()
        self._ax_bottom.clear()
        kind = values.get("kind")
        if kind == "lif_trace":
            self._render_lif(values)
        elif kind == "signal_spikes":
            self._render_signal_spikes(values)
        elif kind == "backprop_convergence":
            self._render_backprop_convergence(values)
        else:
            self._ax_top.text(0.5, 0.5, "(sem sinal para este passo)", ha="center", va="center")
        self._canvas.draw_idle()

    def _render_signal_spikes(self, values: dict[str, object]) -> None:
        signal = np.asarray(values["signal"])
        spikes = np.asarray(values["spikes"])
        n_total = int(values["n_total"])
        level = float(values["level"])
        t = np.arange(len(signal))

        self._ax_top.plot(t, signal, color=SNN_COLOR, linewidth=2)
        if len(t):
            self._ax_top.plot([t[-1]], [signal[-1]], marker="o", markersize=7, color=SNN_COLOR, zorder=4)
        self._ax_top.axhline(level, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="nível de disparo")
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(-1.1, 1.1)
        self._ax_top.set_ylabel("amplitude")
        self._ax_top.legend(loc="upper right", fontsize=8)

        spike_times = t[spikes > 0]
        self._ax_bottom.vlines(spike_times, 0, 1, color=SNN_COLOR, linewidth=2)
        self._ax_bottom.set_xlim(0, n_total)
        self._ax_bottom.set_ylim(0, 1.2)
        self._ax_bottom.set_yticks([])
        self._ax_bottom.set_xlabel("tempo (passos)")
        self._ax_bottom.set_ylabel("spikes")

    def _render_lif(self, values: dict[str, object]) -> None:
        current = np.asarray(values["current"])
        membrane = np.asarray(values["membrane"])
        spikes = np.asarray(values["spikes"])
        v_th = float(values["v_th"])
        t = np.arange(len(membrane))

        self._ax_top.plot(t, current, color=ACCENT_COLOR, linewidth=2)
        if len(t):
            self._ax_top.plot([t[-1]], [current[-1]], marker="o", markersize=6, color=ACCENT_COLOR, zorder=4)
        self._ax_top.set_ylabel("I(t)")
        self._ax_top.set_ylim(-0.05, max(0.5, current.max() * 1.3 if len(current) else 0.5))

        self._ax_bottom.plot(t, membrane, color=SNN_COLOR, linewidth=2)
        if len(t):
            self._ax_bottom.plot([t[-1]], [membrane[-1]], marker="o", markersize=7, color=SNN_COLOR, zorder=4)
        self._ax_bottom.axhline(v_th, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="V_th")
        spike_times = t[spikes > 0]
        if len(spike_times):
            self._ax_bottom.vlines(spike_times, v_th, v_th * 1.25, color=SNN_COLOR, linewidth=2)
        self._ax_bottom.set_ylim(-0.1, v_th * 1.4)
        self._ax_bottom.set_xlabel("tempo (passos)")
        self._ax_bottom.set_ylabel("V(t)")
        self._ax_bottom.legend(loc="upper left", fontsize=8)

    def _render_backprop_convergence(self, values: dict[str, object]) -> None:
        iterations = np.asarray(values["iterations"])
        y = np.asarray(values["y"])
        loss = np.asarray(values["loss"])
        target = float(values["target"])
        n_total = int(values["n_total"])

        self._ax_top.plot(iterations, y, color=CONVERGE_COLOR, linewidth=2, marker="o", markersize=4)
        self._ax_top.plot([iterations[-1]], [y[-1]], marker="o", markersize=9, color=CONVERGE_COLOR, zorder=4)
        self._ax_top.axhline(target, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="alvo")
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(float(values["y_min"]), float(values["y_max"]))
        self._ax_top.set_ylabel("y (saída)")
        self._ax_top.legend(loc="lower right", fontsize=8)

        self._ax_bottom.plot(iterations, loss, color=SNN_COLOR, linewidth=2, marker="o", markersize=4)
        self._ax_bottom.plot([iterations[-1]], [loss[-1]], marker="o", markersize=8, color=SNN_COLOR, zorder=4)
        self._ax_bottom.set_yscale("log")
        self._ax_bottom.set_xlim(0, n_total)
        self._ax_bottom.set_ylim(float(values["loss_min"]), float(values["loss_max"]))
        self._ax_bottom.set_xlabel("iteração")
        self._ax_bottom.set_ylabel("loss (log)")
