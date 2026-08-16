"""Renders single-weight / single-function persistent scenes: a number
line for the scalar quantization demo, the quantization staircase (with a
fading-in annotation), and the SNN surrogate-gradient curve (which morphs
continuously from the useless true derivative into the smooth surrogate).

Each ``render(values)`` call redraws the *same* picture for a given demo,
just with different continuous field values — routing is by the frame's
``kind`` tag, which stays constant across a whole demo, so this widget
never has to guess which layout to build.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.widgets._mpl_perf import fast_clear


class WeightView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 3.2))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, values: dict[str, object]) -> None:
        fast_clear(self._ax)
        kind = values.get("kind")
        if kind == "scalar_quantization":
            self._render_number_line(values)
        elif kind == "staircase":
            self._render_staircase(values)
        elif kind == "quant_derivative":
            self._render_quant_derivative(values)
        elif kind == "surrogate_curve":
            self._render_surrogate_curve(values)
        else:
            self._ax.text(0.5, 0.5, "(sem visualização para este passo)", ha="center", va="center")
            self._ax.axis("off")
        self._canvas.draw_idle()

    # -- scalar_quantization ------------------------------------------
    def _render_number_line(self, values: dict[str, object]) -> None:
        ax = self._ax
        threshold = float(values["threshold"])
        w_display = float(values["w_display"])
        w_quant = values.get("w_quant")

        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1.5, zorder=1)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.18, zorder=0)
        for level in (-1, 0, 1):
            ax.plot([level], [0], marker="|", markersize=24, color=NEUTRAL_COLOR, zorder=2)
            ax.text(level, -0.28, f"{level:+d}", ha="center", fontsize=11)

        marker_color = ACCENT_COLOR if w_quant is not None else BITNET_COLOR
        ax.plot([w_display], [0], marker="o", markersize=16, color=marker_color, zorder=3)
        ax.text(w_display, 0.22, f"{w_display:.2f}", ha="center", fontsize=11, color=marker_color)
        ax.text(threshold, -0.5, f"τ = {threshold:.2f}", ha="left", fontsize=8, color=NEUTRAL_COLOR, style="italic")

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("valor do peso")
        ax.set_title("O peso desliza até o nível quantizado mais próximo")

    # -- backward staircase ---------------------------------------------
    def _render_staircase(self, values: dict[str, object]) -> None:
        ax = self._ax
        w = values["w"]
        q = values["q"]
        threshold = float(values["threshold"])
        annotate = float(values.get("annotate_reveal", 0.0))

        ax.plot(w, q, color=BITNET_COLOR, linewidth=2.5)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.18)
        ax.axvline(threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.axvline(-threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.text(threshold, -1.35, f"τ = {threshold:.2f}", ha="left", fontsize=8, color=NEUTRAL_COLOR, style="italic")
        ax.text(-threshold, -1.35, f"-τ = {-threshold:.2f}", ha="right", fontsize=8, color=NEUTRAL_COLOR, style="italic")
        for level in (-1, 0, 1):
            ax.text(w.min() + 0.05, level, f"Q(w) = {level:+d}", ha="left", va="bottom", fontsize=8, color=BITNET_COLOR)
        ax.set_xlabel("w (peso real)")
        ax.set_ylabel("Q(w)")
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1.5, 1.2)
        ax.set_title("Regiões planas: derivada zero. Saltos: derivada indefinida.")

        if annotate > 0.01:
            ax.annotate(
                "derivada = 0",
                xy=(0.0, 0.0),
                xytext=(0.0, 0.55),
                ha="center",
                fontsize=9,
                color=SNN_COLOR,
                alpha=annotate,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR, alpha=annotate),
            )
            ax.annotate(
                "derivada indefinida",
                xy=(threshold, 0.5),
                xytext=(threshold + 0.35, 0.9),
                ha="left",
                fontsize=9,
                color=SNN_COLOR,
                alpha=annotate,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR, alpha=annotate),
            )

    # -- backward derivative graph (real dQ/dw vs. what STE substitutes) --
    def _render_quant_derivative(self, values: dict[str, object]) -> None:
        ax = self._ax
        w = values["w"]
        curve = values["curve"]
        threshold = float(values["threshold"])
        overlay_reveal = float(values.get("overlay_reveal", 0.0))

        ax.plot(w, curve, color=SNN_COLOR, linewidth=2.5, zorder=3)
        ax.axvline(threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.axvline(-threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.12, zorder=0)
        current_level = float(curve[len(curve) // 2])
        ax.text(w.min() + 0.1, current_level + 0.12, f"dQ/dw = {current_level:.2f}", ha="left", fontsize=9, color=SNN_COLOR)

        if overlay_reveal <= 0.02:
            ax.annotate(
                "indefinida aqui",
                xy=(threshold, 0.0),
                xytext=(threshold + 0.25, 0.55),
                ha="left",
                fontsize=9,
                color=SNN_COLOR,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR),
            )
            ax.annotate(
                "indefinida aqui",
                xy=(-threshold, 0.0),
                xytext=(-threshold - 0.25, 0.55),
                ha="right",
                fontsize=9,
                color=SNN_COLOR,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR),
            )
            ax.set_title("A derivada real de Q(w): zero em toda parte plana")
        else:
            ax.plot(
                w, np.zeros_like(w), color=NEUTRAL_COLOR, linewidth=1.5, linestyle="--",
                alpha=overlay_reveal, label="derivada real (para comparação)", zorder=2,
            )
            ax.legend(loc="center left", fontsize=8)
            ax.set_title("O STE troca essa derivada por uma constante 1 (identidade)")

        ax.set_xlabel("w (peso real)")
        ax.set_ylabel("dQ/dw usada")
        ax.set_xlim(w.min(), w.max())
        ax.set_ylim(-0.3, 1.5)

    # -- surrogate gradient -----------------------------------------------
    def _render_surrogate_curve(self, values: dict[str, object]) -> None:
        ax = self._ax
        x = values["x"]
        bottom_reveal = float(values.get("bottom_reveal", 0.0))
        overlay_reveal = float(values.get("overlay_reveal", 0.0))
        sigmoid_reveal = float(values.get("sigmoid_reveal", 0.0))
        draw_reveal = float(values.get("draw_reveal", 0.0))

        ax.axvline(0, color=NEUTRAL_COLOR, linewidth=1, linestyle=":")

        if bottom_reveal < 0.02:
            spike = values["spike"]
            ax.plot(x, spike, color=SNN_COLOR, linewidth=2.5, label="S(v) — degrau (forward)")
            ax.text(0.05, 1.05, "S(0) = 1", ha="left", fontsize=8, color=SNN_COLOR)
            ax.text(-0.35, -0.1, "S(v<0) = 0", ha="right", fontsize=8, color=SNN_COLOR)
            if sigmoid_reveal > 0.02:
                sigmoid = values["sigmoid"]
                ax.plot(
                    x, sigmoid, color=ACCENT_COLOR, linewidth=2, linestyle="--", alpha=sigmoid_reveal,
                    label="sigmoide suave (antiderivada do gradiente substituto)",
                )
                mid = len(x) // 2
                ax.text(
                    x[mid], float(sigmoid[mid]) + 0.06, f"sigmoide({x[mid]:.1f}) = {float(sigmoid[mid]):.2f}",
                    ha="center", fontsize=7.5, color=ACCENT_COLOR, alpha=sigmoid_reveal,
                )
                ax.legend(loc="lower right", fontsize=7)
            ax.set_ylabel("S(v) — spike (forward)")
            ax.set_title("Função de disparo real, usada no forward")
            ax.set_ylim(-0.2, 1.2)
        elif draw_reveal < 0.02:
            true_derivative = values["true_derivative"]
            ax.plot(x, true_derivative, color=SNN_COLOR, linewidth=2.5, label="gradiente (backward)")
            ax.set_ylabel("gradiente usado no backward")
            ax.set_title("A derivada real: zero em quase todo ponto — inútil para o backward")
        else:
            # the gradient and the sigmoid it comes from are traced together,
            # left to right, up to the same x -- not faded in all at once --
            # so the height of one at the sweep's leading edge is visibly
            # the slope of the other at that same point.
            surrogate = values["surrogate"]
            sigmoid = values["sigmoid"]
            cut = min(len(x), max(2, int(round(draw_reveal * len(x)))))
            xs = x[:cut]
            ax.plot(xs, sigmoid[:cut], color=ACCENT_COLOR, linewidth=2, linestyle="--", label="sigmoide suave")
            ax.plot(xs, surrogate[:cut], color=SNN_COLOR, linewidth=2.5, label="gradiente (backward)")

            tip = cut - 1
            ax.plot([x[tip], x[tip]], [sigmoid[tip], surrogate[tip]], color=NEUTRAL_COLOR, linewidth=1, linestyle=":")
            ax.plot([x[tip]], [sigmoid[tip]], marker="o", markersize=6, color=ACCENT_COLOR, zorder=5)
            ax.plot([x[tip]], [surrogate[tip]], marker="o", markersize=7, color=SNN_COLOR, zorder=5)
            tip_ha = "right" if x[tip] > x[-1] - 0.5 else "center"
            ax.text(
                x[tip], min(1.1, surrogate[tip] + 0.1),
                f"inclinação da sigmoide aqui = altura do gradiente = {surrogate[tip]:.2f}",
                ha=tip_ha, fontsize=7, color=SNN_COLOR,
            )

            peak_idx = int(np.argmax(surrogate))
            if cut > peak_idx and surrogate[peak_idx] > 0.05:
                ax.annotate(
                    f"pico = {surrogate[peak_idx]:.2f}",
                    xy=(x[peak_idx], surrogate[peak_idx]),
                    xytext=(x[peak_idx] + 0.35, surrogate[peak_idx] + 0.15),
                    fontsize=8, color=SNN_COLOR,
                    arrowprops=dict(arrowstyle="->", color=SNN_COLOR),
                )
            if overlay_reveal > 0.02:
                ax.plot(
                    x,
                    values["spike"],
                    color=NEUTRAL_COLOR,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=overlay_reveal,
                    label="spike (forward, para comparação)",
                )
            ax.legend(loc="upper left", fontsize=7)
            ax.set_ylabel("gradiente usado no backward")
            ax.set_title("O gradiente nasce da inclinação da sigmoide, ponto a ponto")
        if bottom_reveal >= 0.02:
            ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("v - v_th")
