"""Renders `live.demo.LiveVowelSnnDemo`'s snapshot: a scrolling multi-row
spike raster (hidden layer, then output layer) on top, and a live
per-vowel confidence bar chart below.

The ONLY demo in this app driven by a live, repeatedly-redrawn snapshot
rather than one `Frame` per render -- see `live/demo.py`'s module
docstring. Still follows every other widget's perf convention
(`fast_clear` + explicit limits every call, see `widgets/_mpl_perf.py`)
and visual convention (`theme.py`'s colors): there was no reason for the
one live widget to look or behave like a different app.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.widgets._mpl_perf import fast_clear


class LiveSpikeView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 3.6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax_top, self._ax_bottom = self._figure.subplots(2, 1, height_ratios=[2, 1])
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

        self._render_raster(snapshot)
        self._render_confidence(snapshot)
        self._canvas.draw_idle()

    def _render_raster(self, snapshot: dict[str, object]) -> None:
        hidden_rows = snapshot["hidden_spike_steps"]
        output_rows = snapshot["output_spike_steps"]
        history_steps = int(snapshot["history_steps"])
        n_hidden, n_output = len(hidden_rows), len(output_rows)

        rows = [[s - history_steps for s in row] for row in hidden_rows + output_rows]
        colors = [SNN_COLOR] * n_hidden + [BITNET_COLOR] * n_output
        if rows:
            self._ax_top.eventplot(rows, colors=colors, linelengths=0.85, linewidths=1.3)
        if n_hidden and n_output:
            self._ax_top.axhline(n_hidden - 0.5, color=NEUTRAL_COLOR, linewidth=1, linestyle="--")
        self._ax_top.set_xlim(-history_steps, 0)
        self._ax_top.set_ylim(-0.6, max(1, n_hidden + n_output) - 0.4)
        self._ax_top.set_yticks([])
        self._ax_top.set_xlabel("passos recentes da SNN (mais à direita = agora)")
        status = "capturando" if snapshot.get("is_capturing") else "parado"
        self._ax_top.set_title(f"Camada oculta ({n_hidden} neurônios) + saída ({n_output}) -- {status}", fontsize=9.5)

    def _render_confidence(self, snapshot: dict[str, object]) -> None:
        names = list(snapshot["class_names"])
        confidences = list(snapshot["class_confidences"])
        predicted = snapshot.get("predicted_class")
        for i, (name, value) in enumerate(zip(names, confidences)):
            color = ACCENT_COLOR if name == predicted else NEUTRAL_COLOR
            self._ax_bottom.add_patch(Rectangle((i - 0.3, 0), 0.6, value, facecolor=color, edgecolor="none", alpha=0.9))
            self._ax_bottom.text(i, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8, weight="bold")
        self._ax_bottom.set_xticks(range(len(names)))
        self._ax_bottom.set_xticklabels([n.upper() for n in names])
        self._ax_bottom.set_xlim(-0.6, len(names) - 0.4)
        self._ax_bottom.set_ylim(0, 1.15)
        self._ax_bottom.set_ylabel("taxa de disparo")
        title = f"Confiança por vogal -- previsto: {predicted.upper()}" if predicted else "Confiança por vogal"
        self._ax_bottom.set_title(title, fontsize=9.5)
