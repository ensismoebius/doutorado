"""The `NeuronView` widget: routes a frame to the renderer for its kind.

Every demo's drawing lives in `widgets/renderers/` — one module per demo,
each a mixin carrying that demo's `_render_*` method plus the constants
only it uses. This file keeps what is genuinely shared: the Qt widget, the
matplotlib figure/canvas it owns, and the dispatch table that turns a
frame's `kind` into a call.

The governing idea of every renderer (enforced by the primitives in
`renderers/_painting.py`): each frame of a demo draws the *same* picture in
the *same* positions -- first the full skeleton (every box and arrow the
demo will ever use, faint and unlabeled, so a first-time viewer already
sees the whole shape of the computation before any number appears), then
the "revealed" content on top of it, with opacity/arrow-fill driven by
continuous 0..1 fields that arrive already smoothly interpolated (see
core/demo.py's tweening). Nothing is ever erased and redrawn as something
unrelated; things only fade or grow in on top of a picture that was
already there.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.widgets._mpl_perf import fast_clear
from efficient_nn_lab.widgets.renderers import (
    ChainLayersRendererMixin,
    ComparisonRendererMixin,
    MatrixAlgebraRendererMixin,
    MlpNetworkRendererMixin,
    PaintingMixin,
    PipelineRenderersMixin,
)


class NeuronView(
    PaintingMixin,
    MlpNetworkRendererMixin,
    MatrixAlgebraRendererMixin,
    ChainLayersRendererMixin,
    PipelineRenderersMixin,
    ComparisonRendererMixin,
    QWidget,
):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6.4, 3.9))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._default_ax_pos = self._ax.get_position()
        # small sigmoid-curve panels (backprop demos) are expensive to
        # create/destroy every animation frame (matplotlib Axes creation is
        # not cheap, and StepPlayer redraws at ~25fps) -- so they are built
        # once, cached by key, and merely cleared + repositioned + shown/
        # hidden on later frames instead of being recreated each time.
        self._inset_axes: dict[str, object] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, values: dict[str, object]) -> None:
        kind = values.get("kind")
        if kind in ("backprop_pipeline", "mlp_network", "forward_pipeline"):
            # shrink the diagram to the left half so the inset panel(s)
            # have clean room on the right instead of floating over the
            # block diagram.
            self._ax.set_position([0.03, 0.06, 0.5, 0.88])
        elif kind == "chain_layers":
            # a full-width ladder (six blocks across, five rows of
            # derivative cards down) with no axis decorations: the default
            # ~77%-of-figure axes box would throw away a quarter of the
            # room the type needs.
            self._ax.set_position(self._CL_AX_RECT)
        elif kind == "comparison_pipeline":
            # the comparison is a text table with no axis decorations at
            # all, so matplotlib's default ~77%-of-figure axes box just
            # throws away a quarter of the width and height that the type
            # could have used. Claim nearly the whole figure: this alone
            # buys ~25% larger text for the same layout.
            self._ax.set_position(self._CMP_AX_RECT)
        else:
            self._ax.set_position(self._default_ax_pos)
        inset_keep = {
            "backprop_pipeline": frozenset(["main"]),
            "mlp_network": frozenset(self._MLP_NAMES),
            "forward_pipeline": frozenset(["forward_numberline"]),
        }.get(kind, frozenset())
        self._hide_insets(keep=inset_keep)
        handler = {
            "matrix_algebra": self._render_matrix_algebra,
            "chain_layers": self._render_chain_layers,
            "mlp_network": self._render_mlp_network,
            "backprop_pipeline": self._render_backprop_pipeline,
            "forward_pipeline": self._render_forward_pipeline,
            "ste_pipeline": self._render_ste_pipeline,
            "guided_pipeline": self._render_guided_pipeline,
            "comparison_pipeline": self._render_comparison_pipeline,
        }.get(kind)
        if handler is None:
            self._reset_axes(xlim=(0, 1), ylim=(0, 1))
            self._ax.text(0.5, 0.5, "(sem diagrama para este passo)", ha="center", va="center")
        else:
            handler(values)
        self._canvas.draw_idle()

