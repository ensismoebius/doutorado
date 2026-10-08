"""The per-vowel confidence bar chart, shared by `LiveSpikeView` (time
raster) and `LiveNodeView` (network diagram) -- both show the same live
classification readout below their own, different, top panel, so this one
function is the single place that draws it instead of two copies drifting
apart.
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from efficient_nn_lab.app.theme import ACCENT_COLOR, NEUTRAL_COLOR


def render_confidence_bars(ax, snapshot: dict[str, object]) -> None:
    names = list(snapshot["class_names"])
    confidences = list(snapshot["class_confidences"])
    predicted = snapshot.get("predicted_class")
    for i, (name, value) in enumerate(zip(names, confidences)):
        color = ACCENT_COLOR if name == predicted else NEUTRAL_COLOR
        ax.add_patch(Rectangle((i - 0.3, 0), 0.6, value, facecolor=color, edgecolor="none", alpha=0.9))
        ax.text(i, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("taxa de disparo")
    title = f"Confiança por vogal -- previsto: {predicted.upper()}" if predicted else "Confiança por vogal"
    ax.set_title(title, fontsize=9.5)
