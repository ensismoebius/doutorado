"""Renders block/schematic "pipeline" scenes: the toy neuron's forward
pass, the STE's dual forward/backward path, the guided BitNet walkthrough,
and the ANN x BitNet x SNN comparison table.

The governing idea: every frame of a given demo draws the *same* picture
in the *same* positions — first the full skeleton (every box and arrow the
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR

_BOX_STYLE = dict(boxstyle="round,pad=0.25", linewidth=1.6)
_SKELETON_ALPHA = 0.28


class NeuronView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6.4, 3.9))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    # -- drawing primitives -----------------------------------------------
    def _reset_axes(self, xlim=(0, 10.5), ylim=(0, 6.2)) -> None:
        self._ax.clear()
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.axis("off")

    def _skeleton_box(self, x: float, y: float, w: float = 1.7, h: float = 0.85) -> None:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor="none", edgecolor=NEUTRAL_COLOR,
            alpha=_SKELETON_ALPHA, linestyle="--", **_BOX_STYLE,
        )
        self._ax.add_patch(patch)

    def _box(self, x: float, y: float, text: str, color: str, alpha: float = 1.0, w: float = 1.7, h: float = 0.85, glow: float = 0.0) -> None:
        if alpha <= 0.02:
            return
        if glow > 0.02:
            glow_patch = FancyBboxPatch(
                (x - w / 2 - 0.12, y - h / 2 - 0.12), w + 0.24, h + 0.24,
                facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.35 * glow, boxstyle="round,pad=0.25",
            )
            self._ax.add_patch(glow_patch)
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor=color, alpha=0.20 * alpha, edgecolor=color, **_BOX_STYLE,
        )
        self._ax.add_patch(patch)
        patch.set_alpha(0.20 * alpha)
        self._ax.text(x, y, text, ha="center", va="center", fontsize=10, alpha=alpha, color="black")

    def _skeleton_arrow(self, p_from: tuple[float, float], p_to: tuple[float, float]) -> None:
        arrow = FancyArrowPatch(
            p_from, p_to, arrowstyle="-|>", mutation_scale=12, color=NEUTRAL_COLOR,
            linewidth=1.0, alpha=_SKELETON_ALPHA, linestyle="--",
        )
        self._ax.add_patch(arrow)

    def _flow_arrow(self, p_from: tuple[float, float], p_to: tuple[float, float], fill: float, color: str = NEUTRAL_COLOR) -> None:
        if fill <= 0.02:
            return
        x0, y0 = p_from
        x1 = x0 + (p_to[0] - x0) * fill
        y1 = y0 + (p_to[1] - y0) * fill
        arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=color, linewidth=2.0, alpha=min(1.0, fill * 1.6))
        self._ax.add_patch(arrow)

    def _fading_text(self, x: float, y: float, text: str, color: str, alpha: float, fontsize: float = 10, weight: str = "normal") -> None:
        if alpha <= 0.02:
            return
        self._ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, alpha=alpha, fontweight=weight)

    # -- dispatch -------------------------------------------------------
    def render(self, values: dict[str, object]) -> None:
        kind = values.get("kind")
        handler = {
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

    # ================================================================
    # backprop/demos/traditional_gd.py — one weight, no quantization: the
    # plain forward (y = w*x) and backward (chain rule) that BitNet's STE
    # and the SNN's surrogate gradient are variations on top of.
    # ================================================================
    _BP_X = (1.2, 5.2)
    _BP_W = (1.2, 3.1)
    _BP_Y = (4.6, 4.15)
    _BP_TARGET = (7.8, 5.2)
    _BP_LOSS = (7.8, 3.1)
    _BP_GRAD_Y = (7.8, 1.1)
    _BP_GRAD_W = (4.6, 1.1)

    def _render_backprop_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(0, 9.4), ylim=(0, 6.2))
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        diff_reveal = float(values["diff_reveal"])
        loss_reveal = float(values["loss_reveal"])
        grady_reveal = float(values["grady_reveal"])
        gradw_reveal = float(values["gradw_reveal"])
        update_reveal = float(values["update_reveal"])
        w_pulse = float(values["w_pulse"])

        for p in (self._BP_X, self._BP_W, self._BP_Y, self._BP_TARGET, self._BP_LOSS, self._BP_GRAD_Y, self._BP_GRAD_W):
            self._skeleton_box(*p)
        self._skeleton_arrow(self._BP_X, self._BP_Y)
        self._skeleton_arrow(self._BP_W, self._BP_Y)
        self._skeleton_arrow(self._BP_Y, self._BP_TARGET)
        self._skeleton_arrow(self._BP_Y, self._BP_LOSS)
        self._skeleton_arrow(self._BP_LOSS, self._BP_GRAD_Y)
        self._skeleton_arrow(self._BP_GRAD_Y, self._BP_GRAD_W)
        self._skeleton_arrow(self._BP_GRAD_W, self._BP_W)

        self._box(*self._BP_X, f"x = {values['x']:g}", NEUTRAL_COLOR)
        self._box(*self._BP_W, f"w = {values['w']:g}", BITNET_COLOR, glow=w_pulse)

        self._flow_arrow(self._BP_X, self._BP_Y, y_reveal, NEUTRAL_COLOR)
        self._flow_arrow(self._BP_W, self._BP_Y, y_reveal, BITNET_COLOR)
        self._box(*self._BP_Y, f"y = {values['y']:g}", CONVERGE_COLOR, alpha=y_reveal)

        self._flow_arrow(self._BP_Y, self._BP_TARGET, target_reveal, NEUTRAL_COLOR)
        self._box(*self._BP_TARGET, f"target = {values['target']:g}", NEUTRAL_COLOR, alpha=target_reveal)
        self._fading_text(
            (self._BP_Y[0] + self._BP_TARGET[0]) / 2 + 0.7, (self._BP_Y[1] + self._BP_TARGET[1]) / 2,
            f"diferença = {values['diff']:g}", ACCENT_COLOR, diff_reveal, fontsize=9,
        )

        self._flow_arrow(self._BP_Y, self._BP_LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._BP_LOSS, f"loss = {values['loss']:g}", SNN_COLOR, alpha=loss_reveal)

        self._flow_arrow(self._BP_LOSS, self._BP_GRAD_Y, grady_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Y, f"dL/dy = {values['grad_y']:g}", SNN_COLOR, alpha=grady_reveal)

        self._flow_arrow(self._BP_GRAD_Y, self._BP_GRAD_W, gradw_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_W, f"dL/dw = {values['grad_w']:g}", SNN_COLOR, alpha=gradw_reveal)
        self._fading_text(4.6, 2.2, "regra da cadeia: x × dL/dy", ACCENT_COLOR, gradw_reveal, fontsize=8)

        self._flow_arrow(self._BP_GRAD_W, self._BP_W, update_reveal, ACCENT_COLOR)
        self._fading_text(1.2, 4.1, f"atualizado -> {values['w_updated']:g}", ACCENT_COLOR, update_reveal, fontsize=9)

    # ================================================================
    # forward.py — inputs -> quantized weights -> sum -> y -> target -> loss
    # ================================================================
    _X1, _X2 = (1.0, 4.7), (1.0, 1.6)
    _W1, _W2 = (3.5, 4.7), (3.5, 1.6)
    _SUM = (6.3, 3.15)
    _Y = (8.5, 3.15)
    _TARGET = (8.5, 5.4)
    _LOSS = (8.5, 0.9)

    def _render_forward_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes()
        x1, x2 = values["x"]
        w1, w2 = values["w_shown"]
        quant_reveal = float(values["quant_reveal"])
        arrow1_fill = float(values["arrow1_fill"])
        arrow2_fill = float(values["arrow2_fill"])
        p1_reveal = float(values["product1_reveal"])
        p2_reveal = float(values["product2_reveal"])
        sum_reveal = float(values["sum_reveal"])
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        diff_reveal = float(values["diff_reveal"])
        loss_reveal = float(values["loss_reveal"])
        h1, h2 = float(values["highlight1"]), float(values["highlight2"])

        # full skeleton, always visible, so the whole pipeline shape is
        # legible from the very first frame
        for p in (self._X1, self._X2, self._W1, self._W2, self._SUM, self._Y, self._TARGET, self._LOSS):
            self._skeleton_box(*p)
        self._skeleton_arrow(self._W1, self._SUM)
        self._skeleton_arrow(self._W2, self._SUM)
        self._skeleton_arrow(self._SUM, self._Y)
        self._skeleton_arrow(self._Y, self._TARGET)
        self._skeleton_arrow(self._Y, self._LOSS)

        self._box(*self._X1, f"x1 = {x1:g}", NEUTRAL_COLOR)
        self._box(*self._X2, f"x2 = {x2:g}", NEUTRAL_COLOR)

        w_label1 = f"Q(w1) = {w1:+.0f}" if quant_reveal >= 0.999 else f"w1 = {w1:.2f}"
        w_label2 = f"Q(w2) = {w2:+.0f}" if quant_reveal >= 0.999 else f"w2 = {w2:.2f}"
        w_color = BITNET_COLOR if quant_reveal > 0.5 else NEUTRAL_COLOR
        self._box(*self._W1, w_label1, w_color, glow=h1)
        self._box(*self._W2, w_label2, w_color, glow=h2)

        self._flow_arrow(self._W1, self._SUM, arrow1_fill, ACCENT_COLOR if h1 > 0.3 else BITNET_COLOR)
        self._flow_arrow(self._W2, self._SUM, arrow2_fill, ACCENT_COLOR if h2 > 0.3 else BITNET_COLOR)
        self._fading_text(4.9, 4.35, f"produto = {values['product1']:g}", ACCENT_COLOR, p1_reveal, fontsize=9)
        self._fading_text(4.9, 1.95, f"produto = {values['product2']:g}", ACCENT_COLOR, p2_reveal, fontsize=9)

        self._box(*self._SUM, "soma", CONVERGE_COLOR, alpha=max(sum_reveal, 0.35 if sum_reveal else 0.0) or 0.01)
        self._fading_text(*self._SUM, "Σ", CONVERGE_COLOR, 1.0 if sum_reveal < 0.02 else 0.0, fontsize=14, weight="bold")

        self._flow_arrow(self._SUM, self._Y, y_reveal, CONVERGE_COLOR)
        self._box(*self._Y, f"y = {values['y']:g}", CONVERGE_COLOR, alpha=y_reveal)

        self._flow_arrow(self._Y, self._TARGET, target_reveal, NEUTRAL_COLOR)
        self._box(*self._TARGET, f"target = {values['target']:g}", NEUTRAL_COLOR, alpha=target_reveal)
        self._fading_text(
            (self._Y[0] + self._TARGET[0]) / 2 + 0.6,
            (self._Y[1] + self._TARGET[1]) / 2,
            f"diferença = {values['diff']:g}",
            ACCENT_COLOR,
            diff_reveal,
            fontsize=9,
        )

        self._flow_arrow(self._Y, self._LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._LOSS, f"loss = {values['loss']:g}", SNN_COLOR, alpha=loss_reveal)

    # ================================================================
    # backward.py — forward path (always) + backward/STE path (reveals)
    # ================================================================
    _FWD_POSITIONS = [(1.2, 4.3), (3.8, 4.3), (6.4, 4.3), (9.0, 4.3)]
    _FWD_LABELS = ["peso real", "quantização", "peso ternário", "operação"]
    _BWD_POSITIONS = [(9.0, 1.6), (6.4, 1.6), (3.8, 1.6), (1.2, 1.6)]
    _BWD_LABELS = ["loss", "gradiente", "STE", "peso real"]

    def _render_ste_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(ylim=(0, 6))
        fwd = float(values["fwd_reveal"])
        bwd = float(values["bwd_reveal"])
        joined = float(values["joined_reveal"])

        for p in self._FWD_POSITIONS + self._BWD_POSITIONS:
            self._skeleton_box(*p, w=2.0)
        for a, b in zip(self._FWD_POSITIONS, self._FWD_POSITIONS[1:]):
            self._skeleton_arrow((a[0] + 1.0, a[1]), (b[0] - 1.0, b[1]))
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._skeleton_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]))

        for pos, label in zip(self._FWD_POSITIONS, self._FWD_LABELS):
            self._box(*pos, label, BITNET_COLOR, alpha=fwd, w=2.0)
        for a, b in zip(self._FWD_POSITIONS, self._FWD_POSITIONS[1:]):
            self._flow_arrow((a[0] + 1.0, a[1]), (b[0] - 1.0, b[1]), fwd, BITNET_COLOR)
        self._fading_text(5.1, 5.35, "FORWARD", BITNET_COLOR, fwd, fontsize=12, weight="bold")

        for pos, label in zip(self._BWD_POSITIONS, self._BWD_LABELS):
            self._box(*pos, label, SNN_COLOR, alpha=bwd, w=2.0)
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._flow_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]), bwd, SNN_COLOR)
        self._fading_text(5.1, 0.65, "BACKWARD (STE)", SNN_COLOR, bwd, fontsize=12, weight="bold")

        # the two paths meet at "peso real": a highlighted ring makes the
        # loop visible once both are on screen.
        self._box(*self._FWD_POSITIONS[0], self._FWD_LABELS[0], BITNET_COLOR, alpha=fwd, w=2.0, glow=joined)
        self._fading_text(5.1, 3.0, "mesma quantização,\ncaminhos diferentes", ACCENT_COLOR, joined, fontsize=9)

    # ================================================================
    # guided_sequence.py — one growing computational graph, 10 steps
    # ================================================================
    _W_POS = (1.2, 3.1)
    _Q_POS = (3.5, 3.1)
    _X_POS = (3.5, 5.2)
    _Y_POS_G = (5.9, 4.15)
    _TARGET_POS_G = (8.3, 5.2)
    _LOSS_POS_G = (8.3, 3.1)
    _GRAD_POS = (5.9, 1.1)

    def _render_guided_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(ylim=(0, 6.4))
        q_reveal = float(values["q_reveal"])
        x_reveal = float(values["x_reveal"])
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        loss_reveal = float(values["loss_reveal"])
        grad_reveal = float(values["grad_reveal"])
        ste_reveal = float(values["ste_reveal"])
        update_reveal = float(values["update_reveal"])
        q_pulse = float(values["q_pulse"])

        for p in (self._W_POS, self._Q_POS, self._X_POS, self._Y_POS_G, self._TARGET_POS_G, self._LOSS_POS_G, self._GRAD_POS):
            self._skeleton_box(*p)
        self._skeleton_arrow(self._W_POS, self._Q_POS)
        self._skeleton_arrow(self._Q_POS, self._Y_POS_G)
        self._skeleton_arrow(self._X_POS, self._Y_POS_G)
        self._skeleton_arrow(self._Y_POS_G, self._TARGET_POS_G)
        self._skeleton_arrow(self._Y_POS_G, self._LOSS_POS_G)
        self._skeleton_arrow(self._LOSS_POS_G, self._GRAD_POS)
        self._skeleton_arrow(self._GRAD_POS, self._W_POS)

        self._box(*self._W_POS, f"w = {values['w_value']:.2f}", BITNET_COLOR)
        self._flow_arrow(self._W_POS, self._Q_POS, q_reveal, BITNET_COLOR)
        self._box(*self._Q_POS, f"Q(w) = {round(values['q_value']):+d}", BITNET_COLOR, alpha=q_reveal, glow=q_pulse)

        self._box(*self._X_POS, f"x = {2.0:g}", NEUTRAL_COLOR, alpha=x_reveal)
        self._flow_arrow(self._Q_POS, self._Y_POS_G, y_reveal, CONVERGE_COLOR)
        self._flow_arrow(self._X_POS, self._Y_POS_G, y_reveal, NEUTRAL_COLOR)
        self._box(*self._Y_POS_G, f"y = {values['y_value']:g}", CONVERGE_COLOR, alpha=y_reveal)

        self._flow_arrow(self._Y_POS_G, self._TARGET_POS_G, target_reveal, NEUTRAL_COLOR)
        self._box(*self._TARGET_POS_G, f"target = {values['target_value']:g}", NEUTRAL_COLOR, alpha=target_reveal)

        self._flow_arrow(self._Y_POS_G, self._LOSS_POS_G, loss_reveal, SNN_COLOR)
        self._box(*self._LOSS_POS_G, f"loss = {values['loss_value']:g}", SNN_COLOR, alpha=loss_reveal)

        self._flow_arrow(self._LOSS_POS_G, self._GRAD_POS, grad_reveal, SNN_COLOR)
        self._box(*self._GRAD_POS, f"dL/dw ~= {values['grad_value']:g}", SNN_COLOR, alpha=grad_reveal)

        self._flow_arrow(self._GRAD_POS, self._W_POS, update_reveal, ACCENT_COLOR, )
        self._fading_text(3.5, 0.5, "STE: gradiente atravessa Q(w) como identidade", ACCENT_COLOR, ste_reveal, fontsize=9)
        self._fading_text(1.2, 4.1, "atualizado", ACCENT_COLOR, update_reveal, fontsize=9)

    # ================================================================
    # comparison — a persistent, growing 3-column table + outputs panel
    # ================================================================
    _ROWS_Y = {
        "Representação": 5.6,
        "Ativação": 4.7,
        "Domínio temporal": 3.8,
        "Treinamento": 2.9,
        "Operação principal": 2.0,
    }
    _COLS_X = {"ANN": 3.0, "BitNet": 6.0, "SNN": 9.0}
    _COL_COLORS = {"ANN": NEUTRAL_COLOR, "BitNet": BITNET_COLOR, "SNN": SNN_COLOR}

    def _render_comparison_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(0, 10.5), ylim=(0, 6.4))
        table = values["table"]
        reveal_map = {
            "Representação": values["reveal_repr"],
            "Ativação": values["reveal_activation"],
            "Domínio temporal": values["reveal_domain"],
            "Treinamento": values["reveal_training"],
            "Operação principal": values["reveal_operation"],
        }

        for col, x in self._COLS_X.items():
            self._fading_text(x, 6.15, col, self._COL_COLORS[col], 1.0, fontsize=12, weight="bold")

        for row_name, y in self._ROWS_Y.items():
            reveal = float(reveal_map[row_name])
            self._fading_text(0.15, y, row_name, "black", max(reveal, 0.12), fontsize=9, weight="bold")
            for (col, x), value in zip(self._COLS_X.items(), table[row_name]):
                self._fading_text(x, y, value, self._COL_COLORS[col], reveal, fontsize=8)

        reveal_outputs = float(values["reveal_outputs"])
        if reveal_outputs > 0.02:
            y = 1.1
            self._fading_text(0.15, y, "Saída", "black", reveal_outputs, fontsize=9, weight="bold")
            self._fading_text(self._COLS_X["ANN"], y, f"y = {values['y_ann']:g}", self._COL_COLORS["ANN"], reveal_outputs, fontsize=9)
            self._fading_text(self._COLS_X["BitNet"], y, f"y = {values['y_bitnet']:g}", self._COL_COLORS["BitNet"], reveal_outputs, fontsize=9)
            self._fading_text(self._COLS_X["SNN"], y, f"{values['snn_spike_count']} spikes", self._COL_COLORS["SNN"], reveal_outputs, fontsize=9)

        reveal_gradients = float(values["reveal_gradients"])
        self._fading_text(
            5.25, 0.55,
            "Tipo de gradiente: exato (ANN) / STE (BitNet) / substituto (SNN) — análogos, não idênticos.",
            ACCENT_COLOR, reveal_gradients, fontsize=8,
        )

        reveal_caveat = float(values["reveal_caveat"])
        self._fading_text(
            5.25, 0.15,
            "Eficiência não é garantida pela arquitetura: depende de hardware, memória e workload.",
            SNN_COLOR, reveal_caveat, fontsize=8, weight="bold",
        )
