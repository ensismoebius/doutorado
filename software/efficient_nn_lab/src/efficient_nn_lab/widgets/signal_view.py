"""Renders time-series demos: raw signal + spike raster, or input current +
membrane potential + spike raster. Both spike_generation and lif_dynamics
reveal their trace progressively as the user steps, so the x-axis is fixed
to the full-length window from the first frame instead of auto-rescaling.
A marker at the trace's current tip makes "where are we right now" always
obvious, especially mid-animation.
"""

from __future__ import annotations

from math import atan2, degrees

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.widgets._mpl_perf import fast_clear


class SignalView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 3.6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax_top, self._ax_bottom = self._figure.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        self._default_top_pos = self._ax_top.get_position()
        self._default_bottom_pos = self._ax_bottom.get_position()
        # cached, not recreated every frame -- see widgets/_mpl_perf.py.
        self._inset_ax = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, values: dict[str, object]) -> None:
        fast_clear(self._ax_top)
        fast_clear(self._ax_bottom)
        if self._inset_ax is not None:
            self._inset_ax.set_visible(False)
        kind = values.get("kind")
        if kind == "backprop_convergence":
            # make room on the right for the sigmoid inset; every other
            # kind keeps the original full-width two-panel layout.
            self._ax_top.set_position([0.11, 0.56, 0.53, 0.38])
            self._ax_bottom.set_position([0.11, 0.13, 0.53, 0.33])
        else:
            self._ax_top.set_position(self._default_top_pos)
            self._ax_bottom.set_position(self._default_bottom_pos)
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
            self._ax_top.text(
                t[-1], signal[-1] + 0.12, f"{signal[-1]:.2f}", ha="center", fontsize=8, color=SNN_COLOR,
            )
        self._ax_top.axhline(level, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label=f"nível de disparo = {level:.2f}")
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(-1.1, 1.1)
        self._ax_top.set_ylabel("amplitude")
        self._ax_top.legend(loc="upper right", fontsize=8)

        spike_times = t[spikes > 0]
        for st in spike_times:
            self._ax_bottom.text(st, 1.05, f"t={st}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.vlines(spike_times, 0, 1, color=SNN_COLOR, linewidth=2)
        self._ax_bottom.set_yscale("linear")  # backprop_convergence leaves this axis log-scaled otherwise
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
            self._ax_top.text(t[-1], current[-1] + 0.03, f"{current[-1]:.2f}", ha="center", fontsize=8, color=ACCENT_COLOR)
        self._ax_top.set_ylabel("I(t)")
        self._ax_top.set_ylim(-0.05, max(0.5, current.max() * 1.3 if len(current) else 0.5))

        self._ax_bottom.plot(t, membrane, color=SNN_COLOR, linewidth=2)
        if len(t):
            self._ax_bottom.plot([t[-1]], [membrane[-1]], marker="o", markersize=7, color=SNN_COLOR, zorder=4)
            self._ax_bottom.text(t[-1], membrane[-1] + 0.05, f"{membrane[-1]:.2f}", ha="center", fontsize=8, color=SNN_COLOR)
        self._ax_bottom.axhline(v_th, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label=f"V_th = {v_th:.2f}")
        spike_times = t[spikes > 0]
        if len(spike_times):
            self._ax_bottom.vlines(spike_times, v_th, v_th * 1.25, color=SNN_COLOR, linewidth=2)
            for st in spike_times:
                self._ax_bottom.text(st, v_th * 1.3, f"t={st}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.set_yscale("linear")  # backprop_convergence leaves this axis log-scaled otherwise
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

        y_min, y_max = float(values["y_min"]), float(values["y_max"])
        self._ax_top.plot(iterations, y, color=CONVERGE_COLOR, linewidth=2, marker="o", markersize=4)
        self._ax_top.plot([iterations[-1]], [y[-1]], marker="o", markersize=9, color=CONVERGE_COLOR, zorder=4)
        self._ax_top.axhline(target, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label=f"alvo = {target:g}")
        # every point gets its value labelled -- only a handful of iterations
        # exist, so this stays readable, unlike the 60-sample traces above.
        label_dy = (y_max - y_min) * 0.05
        for i, (xi, yi) in enumerate(zip(iterations, y)):
            va = "bottom" if i % 2 == 0 else "top"
            dy = label_dy if va == "bottom" else -label_dy
            self._ax_top.text(xi, yi + dy, f"{yi:.2f}", ha="center", va=va, fontsize=7, color=CONVERGE_COLOR)
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(y_min, y_max)
        self._ax_top.set_ylabel("y (saída)")
        self._ax_top.legend(loc="lower right", fontsize=8)

        self._ax_bottom.plot(iterations, loss, color=SNN_COLOR, linewidth=2, marker="o", markersize=4)
        self._ax_bottom.plot([iterations[-1]], [loss[-1]], marker="o", markersize=8, color=SNN_COLOR, zorder=4)
        for xi, li in zip(iterations, loss):
            self._ax_bottom.text(xi, li * 1.3, f"{li:.3f}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.set_yscale("log")
        self._ax_bottom.set_xlim(0, n_total)
        self._ax_bottom.set_ylim(float(values["loss_min"]), float(values["loss_max"]))
        self._ax_bottom.set_xlabel("iteração")
        self._ax_bottom.set_ylabel("loss (log)")

        self._draw_sigmoid_inset(
            rect=(0.68, 0.13, 0.29, 0.78),
            z=float(values["z"]), y=float(y[-1]), slope=float(values["slope"]), grad_z=float(values["grad_z"]),
            z_trail=np.asarray(values["z_trail"]),
        )

    def _draw_sigmoid_inset(
        self, rect: tuple[float, float, float, float], z: float, y: float, slope: float, grad_z: float,
        z_trail: np.ndarray | None = None,
    ) -> None:
        # same construction as neuron_view's inset -- kept local (not
        # shared) since the two widgets have no other coupling and this is
        # the only spot signal_view needs it. Cached rather than recreated
        # every frame -- see widgets/_mpl_perf.py.
        if self._inset_ax is None:
            self._inset_ax = self._figure.add_axes(rect)
        else:
            self._inset_ax.set_position(rect)
            fast_clear(self._inset_ax)
            self._inset_ax.set_visible(True)
        ax = self._inset_ax

        z_grid = np.linspace(-6.0, 6.0, 200)
        ax.plot(z_grid, 1.0 / (1.0 + np.exp(-z_grid)), color=CONVERGE_COLOR, linewidth=2)
        ax.axvline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")

        # every past iteration's point stays marked on the curve -- a new
        # dot is added each iteration, exactly like the y-vs-iteration
        # chart to its left, instead of one dot relocating and erasing
        # where it has already been.
        if z_trail is not None and len(z_trail) > 1:
            y_trail = 1.0 / (1.0 + np.exp(-z_trail[:-1]))
            ax.plot(z_trail[:-1], y_trail, marker="o", markersize=5, color=CONVERGE_COLOR, alpha=0.4, linestyle="None", zorder=3)

        ax.plot([z], [y], marker="o", markersize=9, color=CONVERGE_COLOR, zorder=4)
        ax.text(z, y + 0.08, f"y = {y:.2f}", ha="center", fontsize=7.5, color=CONVERGE_COLOR)

        half = 2.0
        z_tan = np.array([z - half, z + half])
        y_tan = y + slope * (z_tan - z)
        ax.plot(z_tan, y_tan, color=ACCENT_COLOR, linewidth=1.5, linestyle="--")
        ax.text(z_tan[0], y_tan[0], f"σ'(z) = {slope:.2f}", ha="right", va="top", fontsize=7, color=ACCENT_COLOR)

        direction = -1.0 if grad_z >= 0 else 1.0
        dz = direction * 0.9
        dy = slope * dz
        # cheap shaft + rotated-triangle arrowhead, not ax.annotate's
        # FancyArrowPatch -- see neuron_view._flow_arrow for why.
        ax.plot([z, z + dz], [y, y + dy], color=SNN_COLOR, linewidth=2)
        angle = degrees(atan2(dy, dz)) - 90.0
        ax.plot([z + dz], [y + dy], marker=(3, 0, angle), markersize=10, color=SNN_COLOR, linestyle="None")
        ax.text(
            z + dz, y + dy + (0.1 if direction > 0 else -0.15),
            "descida do gradiente", ha="center", fontsize=7, color=SNN_COLOR,
        )

        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("z", fontsize=8)
        ax.set_ylabel("σ(z)", fontsize=8)
        ax.set_title("Ativação: onde estamos na curva", fontsize=8.5)
        ax.tick_params(labelsize=7)
