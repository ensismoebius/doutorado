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
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.backprop.demos.rube_goldberg_physics import W2_DRAG_BOUNDS, X_DRAG_BOUNDS
from efficient_nn_lab.widgets._mpl_perf import fast_clear
from efficient_nn_lab.widgets.renderers import (
    AutoencoderComparisonRendererMixin,
    ChainLayersRendererMixin,
    ComparisonRendererMixin,
    MatrixAlgebraRendererMixin,
    MlpNetworkRendererMixin,
    NormalizationRendererMixin,
    PaintingMixin,
    PipelineRenderersMixin,
    RubeGoldbergRendererMixin,
    TimestepsRendererMixin,
)


class NeuronView(
    PaintingMixin,
    MlpNetworkRendererMixin,
    MatrixAlgebraRendererMixin,
    ChainLayersRendererMixin,
    RubeGoldbergRendererMixin,
    PipelineRenderersMixin,
    ComparisonRendererMixin,
    TimestepsRendererMixin,
    AutoencoderComparisonRendererMixin,
    NormalizationRendererMixin,
    QWidget,
):
    #: Emitted on a canvas click while the `rube_goldberg` kind is the one
    #: currently rendered, AND the click did not start on a draggable
    #: handle -- see `_on_canvas_press`/`_on_canvas_release`. Every other
    #: demo is driven exclusively by the sidebar transport; this is the one
    #: exception, wired by main_window.py to the exact same "advance to the
    #: next checkpoint" call the "Próximo" button makes, so a click is
    #: literally "push the machine", not a second code path.
    advance_requested = Signal()
    #: Emitted once, on release, after dragging the lever (sets "w2") or
    #: the roldana (sets "x") to a new value -- `(name, value)`, the exact
    #: same shape as `ControlsWidget.parameter_changed`, so main_window.py
    #: wires it to the SAME `_on_parameter_changed` handler a slider uses
    #: instead of inventing a second update path. "Edit between runs": no
    #: signal fires until release, so the machine only ever re-simulates
    #: once per drag, never on every intermediate mouse-move.
    parameter_drag_committed = Signal(str, float)

    #: Vertical data-units of drag that sweep a handle's full value range --
    #: shared between the live preview (motion) and the committed value
    #: (release) so they never disagree about what a given drag means.
    _DRAG_SPAN = 3.0
    _DRAG_BOUNDS = {"w2": W2_DRAG_BOUNDS, "x": X_DRAG_BOUNDS}

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
        self._current_kind: str | None = None
        self._last_values: dict[str, object] = {}
        # drag state for backprop.rube_goldberg's two handles (lever/w2,
        # roldana/x) -- None whenever no drag is in progress.
        self._rg_drag_param: str | None = None
        self._rg_drag_press_y: float = 0.0
        self._rg_drag_start_value: float = 0.0
        self._rg_drag_preview_artist = None
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def _on_canvas_press(self, event: object) -> None:
        if self._current_kind != "rube_goldberg":
            return
        xdata, ydata = getattr(event, "xdata", None), getattr(event, "ydata", None)
        param = self.rg_handle_at(self._last_values, xdata, ydata) if xdata is not None else None
        if param is None:
            self._rg_drag_param = None
            return
        chain_values = self._last_values.get("chain_values", (0.0,) * 5)
        self._rg_drag_param = param
        self._rg_drag_press_y = ydata
        self._rg_drag_start_value = float(chain_values[2] if param == "w2" else chain_values[4])

    def _on_canvas_motion(self, event: object) -> None:
        if self._rg_drag_param is None:
            return
        ydata = getattr(event, "ydata", None)
        if ydata is None:
            return
        value = self._rg_drag_value(self._rg_drag_param, ydata)
        text = f"w2 = {value:+.2f}" if self._rg_drag_param == "w2" else f"x = {value:+.2f}"
        if self._rg_drag_preview_artist is None:
            self._rg_drag_preview_artist = self._ax.text(
                0.5, 1.02, text, transform=self._ax.transAxes, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333",
            )
        else:
            self._rg_drag_preview_artist.set_text(text)
        self._canvas.draw_idle()

    def _on_canvas_release(self, event: object) -> None:
        param = self._rg_drag_param
        self._rg_drag_param = None
        if self._rg_drag_preview_artist is not None:
            self._rg_drag_preview_artist.remove()
            self._rg_drag_preview_artist = None
            self._canvas.draw_idle()
        if param is not None:
            ydata = getattr(event, "ydata", None)
            value = self._rg_drag_value(param, ydata) if ydata is not None else self._rg_drag_start_value
            self.parameter_drag_committed.emit(param, value)
            return
        if self._current_kind == "rube_goldberg":
            self.advance_requested.emit()

    def _rg_drag_value(self, param: str, ydata: float) -> float:
        lo, hi = self._DRAG_BOUNDS[param]
        delta = (ydata - self._rg_drag_press_y) / self._DRAG_SPAN * (hi - lo)
        return max(lo, min(hi, self._rg_drag_start_value + delta))

    def render(self, values: dict[str, object]) -> None:
        kind = values.get("kind")
        self._current_kind = kind if isinstance(kind, str) else None
        self._last_values = values
        # fast_clear() below wipes every artist off the axes, including any
        # live drag-preview text -- drop the now-dangling reference so a
        # later motion event recreates it instead of calling .set_text() on
        # an artist that's no longer attached to anything.
        self._rg_drag_preview_artist = None
        if self._current_kind != "rube_goldberg":
            # switching away from the demo mid-drag (e.g. the sidebar tree)
            # must not leave a stale drag armed for whatever kind renders
            # next -- a release on an unrelated demo would otherwise still
            # read the old _rg_drag_param and emit a bogus parameter change.
            self._rg_drag_param = None
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
        elif kind == "autoencoder_comparison_pipeline":
            # its insets and bottleneck sketch are laid out in figure
            # coordinates: a whole-figure axes makes data coords == figure coords.
            self._ax.set_position(self._AEC_AX_RECT)
        else:
            self._ax.set_position(self._default_ax_pos)
        inset_keep = {
            "backprop_pipeline": frozenset(["main"]),
            "mlp_network": frozenset(self._MLP_NAMES),
            "forward_pipeline": frozenset(["forward_numberline"]),
            "autoencoder_comparison_pipeline": frozenset(["aec_signal", "aec_error"]),
        }.get(kind, frozenset())
        self._hide_insets(keep=inset_keep)
        handler = {
            "matrix_algebra": self._render_matrix_algebra,
            "chain_layers": self._render_chain_layers,
            "rube_goldberg": self._render_rube_goldberg,
            "mlp_network": self._render_mlp_network,
            "backprop_pipeline": self._render_backprop_pipeline,
            "forward_pipeline": self._render_forward_pipeline,
            "ste_pipeline": self._render_ste_pipeline,
            "guided_pipeline": self._render_guided_pipeline,
            "comparison_pipeline": self._render_comparison_pipeline,
            "timesteps_tensor": self._render_timesteps_tensor,
            "autoencoder_comparison_pipeline": self._render_autoencoder_comparison_pipeline,
            "normalization_pipeline": self._render_normalization_pipeline,
        }.get(kind)
        if handler is None:
            self._reset_axes(xlim=(0, 1), ylim=(0, 1))
            self._ax.text(0.5, 0.5, "(sem diagrama para este passo)", ha="center", va="center")
        else:
            handler(values)
        self._canvas.draw_idle()

