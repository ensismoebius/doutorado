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

from math import atan2, degrees

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.backprop.activation import sigmoid_derivative
from efficient_nn_lab.widgets._mpl_perf import fast_clear

_BOX_STYLE = dict(boxstyle="round,pad=0.25", linewidth=2.4)
_SKELETON_ALPHA = 0.35
_FILL_ALPHA = 0.55


class NeuronView(QWidget):
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

    # -- drawing primitives -----------------------------------------------
    def _reset_axes(self, xlim=(0, 10.5), ylim=(0, 6.2)) -> None:
        fast_clear(self._ax)
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.axis("off")

    def _skeleton_box(self, x: float, y: float, w: float = 1.7, h: float = 0.85) -> None:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor="none", edgecolor=NEUTRAL_COLOR,
            alpha=_SKELETON_ALPHA, linestyle="--", **_BOX_STYLE,
        )
        self._ax.add_patch(patch)

    def _box(self, x: float, y: float, text: str, color: str, alpha: float = 1.0, w: float = 1.7, h: float = 0.85, glow: float = 0.0, fontsize: float = 10) -> None:
        if alpha <= 0.02:
            return
        if glow > 0.02:
            glow_patch = FancyBboxPatch(
                (x - w / 2 - 0.12, y - h / 2 - 0.12), w + 0.24, h + 0.24,
                facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.5 * glow, boxstyle="round,pad=0.25",
            )
            self._ax.add_patch(glow_patch)
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor=color, alpha=_FILL_ALPHA * alpha, edgecolor=color, **_BOX_STYLE,
        )
        self._ax.add_patch(patch)
        patch.set_alpha(_FILL_ALPHA * alpha)
        self._ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, alpha=alpha, color="black")

    def _skeleton_arrow(self, p_from: tuple[float, float], p_to: tuple[float, float]) -> None:
        # a plain dashed line, not a FancyArrowPatch: skeleton arrows are
        # faint background hints (never the focus), drawn in bulk every
        # single frame, and FancyArrowPatch's arrowhead-clipping math
        # (Bezier intersection against the mutation path) turned out to be
        # the single biggest cost in the multilayer-network diagram's
        # frame time -- a plain Line2D is a fraction of the cost and reads
        # just as clearly at this alpha/linewidth.
        self._ax.plot(
            [p_from[0], p_to[0]], [p_from[1], p_to[1]], color=NEUTRAL_COLOR,
            linewidth=1.0, alpha=_SKELETON_ALPHA, linestyle="--", solid_capstyle="butt",
        )

    def _flow_arrow(self, p_from: tuple[float, float], p_to: tuple[float, float], fill: float, color: str = NEUTRAL_COLOR) -> None:
        if fill <= 0.02:
            return
        # shaft (plain line) + a rotated triangle marker for the arrowhead,
        # instead of FancyArrowPatch: same reasoning as _skeleton_arrow --
        # FancyArrowPatch's Bezier-based arrowhead clipping was the
        # dominant per-frame cost in diagrams with many arrows (the
        # multilayer-network demo draws a dozen+ every frame).
        x0, y0 = p_from
        x1 = x0 + (p_to[0] - x0) * fill
        y1 = y0 + (p_to[1] - y0) * fill
        alpha = min(1.0, fill * 1.6)
        self._ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.0, alpha=alpha, solid_capstyle="butt")
        angle = degrees(atan2(y1 - y0, x1 - x0)) - 90.0
        self._ax.plot([x1], [y1], marker=(3, 0, angle), markersize=11, color=color, alpha=alpha, linestyle="None")

    def _fading_text(self, x: float, y: float, text: str, color: str, alpha: float, fontsize: float = 10, weight: str = "normal") -> None:
        if alpha <= 0.02:
            return
        self._ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, alpha=alpha, fontweight=weight)

    def _equation_near(self, x: float, y: float, text: str, alpha: float, dy: float = -0.62) -> None:
        """A small formula placed right under the box whose value it explains.

        Kept visually distinct (italic, grey, smaller) from the box's own
        label so it always reads as "this is the rule", not another value.
        """
        if alpha <= 0.02:
            return
        self._ax.text(
            x, y + dy, text, ha="center", va="center", fontsize=7.5, color=NEUTRAL_COLOR,
            alpha=alpha, style="italic",
        )

    # -- cached inset-axes pool (see __init__ note on why this is cached
    # instead of created fresh every frame) ---------------------------------
    def _get_inset(self, key: str, rect: tuple[float, float, float, float]):
        ax = self._inset_axes.get(key)
        if ax is None:
            ax = self._figure.add_axes(rect)
            self._inset_axes[key] = ax
        else:
            ax.set_position(rect)
            fast_clear(ax)
            ax.set_visible(True)
        return ax

    def _hide_insets(self, keep: frozenset[str] = frozenset()) -> None:
        for key, ax in self._inset_axes.items():
            if key not in keep:
                ax.set_visible(False)

    # -- a small sigmoid-curve panel: point on the curve, tangent line
    # (slope = the local derivative), and an arrow showing which way
    # gradient descent pushes z. Used both by the single-neuron backprop
    # demo and, one per neuron, by the 4-layer network demo below.
    def _paint_sigmoid(
        self, ax, z: float, y: float, slope: float, grad_z: float, point_reveal: float,
        tangent_reveal: float, arrow_reveal: float, title: str, compact: bool = False,
    ) -> None:
        z_grid = np.linspace(-6.0, 6.0, 80 if compact else 200)
        ax.plot(z_grid, 1.0 / (1.0 + np.exp(-z_grid)), color=CONVERGE_COLOR, linewidth=2 if compact else 2.2)
        ax.axvline(0, color=NEUTRAL_COLOR, linewidth=0.7, linestyle=":")

        if point_reveal > 0.02:
            ax.plot([z], [y], marker="o", markersize=7 if compact else 9, color=CONVERGE_COLOR, alpha=point_reveal, zorder=4)
            if not compact:
                ax.text(z, y + 0.08, f"y = σ({z:.2f}) = {y:.2f}", ha="center", fontsize=7.5, color=CONVERGE_COLOR, alpha=point_reveal)

        if tangent_reveal > 0.02:
            half = 2.0
            z_tan = np.array([z - half, z + half])
            y_tan = y + slope * (z_tan - z)
            ax.plot(z_tan, y_tan, color=ACCENT_COLOR, linewidth=1.3 if compact else 1.5, linestyle="--", alpha=tangent_reveal)
            if not compact:
                ax.text(
                    z_tan[0], y_tan[0], f"σ'(z) = {slope:.2f}", ha="right", va="top",
                    fontsize=7, color=ACCENT_COLOR, alpha=tangent_reveal,
                )

        if arrow_reveal > 0.02:
            direction = -1.0 if grad_z >= 0 else 1.0
            dz = direction * 0.9
            dy = slope * dz
            ax.annotate(
                "", xy=(z + dz, y + dy), xytext=(z, y),
                arrowprops=dict(arrowstyle="-|>", color=SNN_COLOR, linewidth=1.6 if compact else 2, alpha=arrow_reveal),
            )
            if not compact:
                ax.text(
                    z + dz, y + dy + (0.1 if direction > 0 else -0.15),
                    "descida do gradiente", ha="center", fontsize=7, color=SNN_COLOR, alpha=arrow_reveal,
                )

        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-0.15, 1.15)
        if compact:
            ax.set_title(title, fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("z", fontsize=8)
            ax.set_ylabel("σ(z)", fontsize=8)
            ax.set_title(title, fontsize=8.5)
            ax.tick_params(labelsize=7)

    # -- forward.py's companion panel: *why* Q(w1) and Q(w2) come out the
    # way they do, geometrically -- where each real weight actually sits
    # relative to the +-tau dead zone, sliding to its quantized level once
    # revealed (mirrors bitnet/demos/scalar_quantization.py's own slide).
    def _paint_weight_numberline(
        self, ax, w1: float, w2: float, w1q: int, w2q: int, threshold: float, reveal1: float, reveal2: float,
    ) -> None:
        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1.3, zorder=1)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.18, zorder=0)
        for level in (-1, 0, 1):
            ax.plot([level], [0], marker="|", markersize=20, color=NEUTRAL_COLOR, zorder=2)
            ax.text(level, -0.22, f"{level:+d}", ha="center", fontsize=9)
        ax.text(threshold, -0.42, f"τ = {threshold:.2f}", ha="left", fontsize=7.5, color=NEUTRAL_COLOR, style="italic")
        ax.text(-threshold, -0.42, f"-τ = {-threshold:.2f}", ha="right", fontsize=7.5, color=NEUTRAL_COLOR, style="italic")

        def draw(w: float, wq: int, reveal: float, row_y: float, label: str) -> None:
            reveal = max(0.0, min(1.0, reveal))
            display = w + (wq - w) * reveal
            color = ACCENT_COLOR if reveal >= 0.999 else BITNET_COLOR
            ax.plot([display], [row_y], marker="o", markersize=12, color=color, zorder=4)
            text = f"Q({label}) = {round(wq):+d}" if reveal >= 0.999 else f"{label} = {w:.2f}"
            ax.text(display, row_y + 0.14, text, ha="center", fontsize=8, color=color)

        draw(w1, w1q, reveal1, 0.28, "w1")
        draw(w2, w2q, reveal2, 0.58, "w2")

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.55, 0.85)
        ax.set_yticks([])
        ax.set_xlabel("valor do peso", fontsize=8)
        ax.set_title("Onde w1 e w2 caem em relação a ±τ", fontsize=8.5)
        ax.tick_params(labelsize=7)

    # -- dispatch -------------------------------------------------------
    def render(self, values: dict[str, object]) -> None:
        kind = values.get("kind")
        if kind in ("backprop_pipeline", "mlp_network", "forward_pipeline"):
            # shrink the diagram to the left half so the inset panel(s)
            # have clean room on the right instead of floating over the
            # block diagram.
            self._ax.set_position([0.03, 0.06, 0.5, 0.88])
        else:
            self._ax.set_position(self._default_ax_pos)
        inset_keep = {
            "backprop_pipeline": frozenset(["main"]),
            "mlp_network": frozenset(self._MLP_NAMES),
            "forward_pipeline": frozenset(["forward_numberline"]),
        }.get(kind, frozenset())
        self._hide_insets(keep=inset_keep)
        handler = {
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

    # ================================================================
    # backprop/demos/multilayer_network.py — 3 -> 2 -> 2 -> 1, sigmoid
    # everywhere. Walked one neuron at a time (forward left-to-right, then
    # backward right-to-left); the "active" neuron gets a glow, its
    # incoming edges get weight labels, and the detail panel focuses on it.
    # Unlike the single-neuron demo, every neuron keeps its *own* small
    # sigmoid panel, all visible at once (not one shared/switching inset)
    # so students can watch each neuron's point/tangent/gradient evolve
    # independently as the walkthrough proceeds.
    # ================================================================
    _MLP_X = [(0.6, 5.6), (0.6, 3.5), (0.6, 1.4)]
    _MLP_L1 = [(3.0, 4.7), (3.0, 2.3)]
    _MLP_L2 = [(5.4, 4.7), (5.4, 2.3)]
    _MLP_O = (7.8, 3.5)
    _MLP_TARGET = (7.8, 5.6)
    _MLP_LOSS = (7.8, 1.4)
    _MLP_NAMES = ["L1-A", "L1-B", "L2-C", "L2-D", "Saída"]
    _MLP_POS = {"L1-A": _MLP_L1[0], "L1-B": _MLP_L1[1], "L2-C": _MLP_L2[0], "L2-D": _MLP_L2[1], "Saída": _MLP_O}
    _MLP_INPUT_POS = {
        "L1-A": _MLP_X, "L1-B": _MLP_X,
        "L2-C": _MLP_L1, "L2-D": _MLP_L1,
        "Saída": _MLP_L2,
    }
    # figure-fraction rects for the 5 per-neuron sigmoid panels, arranged
    # in the same left-to-right layer order as the diagram itself (L1
    # column, L2 column, output column) so "which graph is which neuron"
    # is obvious without reading labels.
    _MLP_INSET_RECTS = {
        "L1-A": (0.55, 0.56, 0.13, 0.38),
        "L1-B": (0.55, 0.08, 0.13, 0.38),
        "L2-C": (0.705, 0.56, 0.13, 0.38),
        "L2-D": (0.705, 0.08, 0.13, 0.38),
        "Saída": (0.86, 0.32, 0.13, 0.38),
    }

    def _render_mlp_network(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(-0.1, 8.6), ylim=(-3.0, 6.6))
        x, target = values["x"], float(values["target"])
        w1, w2, w3 = values["w1"], values["w2"], values["w3"]
        z1, y1, z2, y2, zO, yO = values["z1"], values["y1"], values["z2"], values["y2"], float(values["zO"]), float(values["yO"])
        gz1, gz2, gzO = values["grad_z1"], values["grad_z2"], float(values["grad_zO"])
        fwd = {
            "L1-A": float(values["fwd_l1a"]), "L1-B": float(values["fwd_l1b"]),
            "L2-C": float(values["fwd_l2c"]), "L2-D": float(values["fwd_l2d"]), "Saída": float(values["fwd_o"]),
        }
        bwd = {
            "L1-A": float(values["bwd_l1a"]), "L1-B": float(values["bwd_l1b"]),
            "L2-C": float(values["bwd_l2c"]), "L2-D": float(values["bwd_l2d"]), "Saída": float(values["bwd_o"]),
        }
        y_val = {"L1-A": float(y1[0]), "L1-B": float(y1[1]), "L2-C": float(y2[0]), "L2-D": float(y2[1]), "Saída": yO}
        gz_val = {"L1-A": float(gz1[0]), "L1-B": float(gz1[1]), "L2-C": float(gz2[0]), "L2-D": float(gz2[1]), "Saída": gzO}
        w_row = {"L1-A": w1[0], "L1-B": w1[1], "L2-C": w2[0], "L2-D": w2[1], "Saída": w3[0]}
        loss_reveal = float(values["loss_reveal"])
        update_reveal = float(values["update_reveal"])
        active = values.get("active", "")

        # full skeleton: every box and every weighted edge, from frame one.
        for p in self._MLP_X + self._MLP_L1 + self._MLP_L2 + [self._MLP_O, self._MLP_TARGET, self._MLP_LOSS]:
            self._skeleton_box(*p, w=1.4, h=0.75)
        for name in self._MLP_NAMES:
            for p_in in self._MLP_INPUT_POS[name]:
                self._skeleton_arrow(p_in, self._MLP_POS[name])
        self._skeleton_arrow(self._MLP_O, self._MLP_TARGET)
        self._skeleton_arrow(self._MLP_O, self._MLP_LOSS)

        for i, p in enumerate(self._MLP_X):
            self._box(*p, f"x{i+1} = {x[i]:g}", NEUTRAL_COLOR, w=1.4, h=0.75, fontsize=9)

        for name in self._MLP_NAMES:
            r = fwd[name]
            if r <= 0.02:
                continue
            pos = self._MLP_POS[name]
            glow = 1.0 if active == name else 0.0
            for p_in in self._MLP_INPUT_POS[name]:
                self._flow_arrow(p_in, pos, r, ACCENT_COLOR if glow else BITNET_COLOR)
            text = f"{name}\ny = {y_val[name]:.2f}"
            if bwd[name] > 0.02:
                text += f"\ndL/dz={gz_val[name]:.3f}"
            self._box(*pos, text, CONVERGE_COLOR if bwd[name] > 0.02 else BITNET_COLOR, alpha=r, w=1.4, h=0.75, glow=glow, fontsize=8)
            if glow:
                self._equation_near(*pos, "z = Σ w·entrada;  y = σ(z)", r, dy=-0.55)

        self._flow_arrow(self._MLP_O, self._MLP_TARGET, loss_reveal, NEUTRAL_COLOR)
        self._box(*self._MLP_TARGET, f"alvo = {target:g}", NEUTRAL_COLOR, alpha=loss_reveal, w=1.4, h=0.75, fontsize=9)
        self._flow_arrow(self._MLP_O, self._MLP_LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._MLP_LOSS, f"loss = {float(values['loss']):.3f}", SNN_COLOR, alpha=loss_reveal, w=1.4, h=0.75, fontsize=9)

        # backward: a thin orange return-edge drawn *behind* each active
        # neuron's incoming connections, showing gradient flowing the
        # opposite way along the same wires.
        for name in self._MLP_NAMES:
            if bwd[name] <= 0.02:
                continue
            for p_in in self._MLP_INPUT_POS[name]:
                self._flow_arrow(self._MLP_POS[name], p_in, bwd[name] * 0.6, SNN_COLOR)

        detail = str(values.get("active_detail", ""))
        if detail:
            self._ax.text(
                0.1, -1.1, detail, ha="left", va="top", fontsize=7.5, color="black",
                family="monospace", linespacing=1.6,
            )

        if update_reveal > 0.02:
            self._fading_text(4.2, -2.3, "todos os pesos atualizados: w ← w - taxa · dL/dw", ACCENT_COLOR, update_reveal, fontsize=8)

        z_val = {"L1-A": float(z1[0]), "L1-B": float(z1[1]), "L2-C": float(z2[0]), "L2-D": float(z2[1]), "Saída": zO}
        slope_val = {name: float(sigmoid_derivative(z_val[name])) for name in self._MLP_NAMES}
        for name in self._MLP_NAMES:
            ax = self._get_inset(name, self._MLP_INSET_RECTS[name])
            self._paint_sigmoid(
                ax, z=z_val[name], y=y_val[name], slope=slope_val[name], grad_z=gz_val[name],
                point_reveal=fwd[name], tangent_reveal=fwd[name], arrow_reveal=bwd[name],
                title=name, compact=True,
            )

    # ================================================================
    # backprop/demos/traditional_gd.py — one weight, sigmoid activation: the
    # forward (z = w*x, y = sigma(z)) and backward (three-link chain rule)
    # that BitNet's STE and the SNN's surrogate gradient are variations on.
    # A sigmoid-curve inset (point + tangent + descent direction) shares
    # the figure with the block diagram — see _paint_sigmoid.
    # ================================================================
    _BP_X = (0.7, 5.3)
    _BP_W = (0.7, 3.0)
    _BP_Z = (3.0, 4.15)
    _BP_Y = (5.6, 4.15)
    _BP_TARGET = (5.6, 6.15)
    _BP_LOSS = (5.6, 2.15)
    _BP_GRAD_Y = (3.0, 1.1)
    _BP_GRAD_Z = (5.6, 0.15)
    _BP_GRAD_W = (0.7, 1.1)

    def _render_backprop_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(-0.2, 6.6), ylim=(-0.3, 6.7))
        z_reveal = float(values["z_reveal"])
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        diff_reveal = float(values["diff_reveal"])
        loss_reveal = float(values["loss_reveal"])
        grady_reveal = float(values["grady_reveal"])
        gradz_reveal = float(values["gradz_reveal"])
        gradw_reveal = float(values["gradw_reveal"])
        update_reveal = float(values["update_reveal"])
        w_pulse = float(values["w_pulse"])

        boxes = (
            self._BP_X, self._BP_W, self._BP_Z, self._BP_Y, self._BP_TARGET, self._BP_LOSS,
            self._BP_GRAD_Y, self._BP_GRAD_Z, self._BP_GRAD_W,
        )
        for p in boxes:
            self._skeleton_box(*p, w=1.5)
        self._skeleton_arrow(self._BP_X, self._BP_Z)
        self._skeleton_arrow(self._BP_W, self._BP_Z)
        self._skeleton_arrow(self._BP_Z, self._BP_Y)
        self._skeleton_arrow(self._BP_Y, self._BP_TARGET)
        self._skeleton_arrow(self._BP_Y, self._BP_LOSS)
        self._skeleton_arrow(self._BP_LOSS, self._BP_GRAD_Y)
        self._skeleton_arrow(self._BP_GRAD_Y, self._BP_GRAD_Z)
        self._skeleton_arrow(self._BP_GRAD_Z, self._BP_GRAD_W)
        self._skeleton_arrow(self._BP_GRAD_W, self._BP_W)

        self._box(*self._BP_X, f"x = {values['x']:g}", NEUTRAL_COLOR, w=1.5)
        self._box(*self._BP_W, f"w = {values['w']:g}", BITNET_COLOR, w=1.5, glow=w_pulse)

        self._flow_arrow(self._BP_X, self._BP_Z, z_reveal, NEUTRAL_COLOR)
        self._flow_arrow(self._BP_W, self._BP_Z, z_reveal, BITNET_COLOR)
        self._box(*self._BP_Z, f"z = {values['z']:g}", BITNET_COLOR, alpha=z_reveal, w=1.5)
        self._equation_near(*self._BP_Z, "z = w · x", z_reveal, dy=-0.55)

        self._flow_arrow(self._BP_Z, self._BP_Y, y_reveal, CONVERGE_COLOR)
        self._box(*self._BP_Y, f"y = σ(z)\n= {values['y']:.3f}", CONVERGE_COLOR, alpha=y_reveal, w=1.9)
        self._equation_near(*self._BP_Y, "y = σ(z) = 1/(1+e⁻ᶻ)", y_reveal, dy=0.6)

        self._flow_arrow(self._BP_Y, self._BP_TARGET, target_reveal, NEUTRAL_COLOR)
        self._box(*self._BP_TARGET, f"target = {values['target']:g}", NEUTRAL_COLOR, alpha=target_reveal, w=1.5)
        self._fading_text(
            (self._BP_Y[0] + self._BP_TARGET[0]) / 2 - 1.05, (self._BP_Y[1] + self._BP_TARGET[1]) / 2,
            f"diferença = {values['diff']:.3f}", ACCENT_COLOR, diff_reveal, fontsize=8,
        )

        self._flow_arrow(self._BP_Y, self._BP_LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._BP_LOSS, f"loss = {values['loss']:.3f}", SNN_COLOR, alpha=loss_reveal, w=1.5)
        self._equation_near(*self._BP_LOSS, "L = ½ (y - target)²", loss_reveal, dy=-0.55)

        self._flow_arrow(self._BP_LOSS, self._BP_GRAD_Y, grady_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Y, f"dL/dy = {values['grad_y']:.2f}", SNN_COLOR, alpha=grady_reveal, w=1.5)
        self._equation_near(*self._BP_GRAD_Y, "dL/dy = y - target", grady_reveal, dy=-0.55)

        self._flow_arrow(self._BP_GRAD_Y, self._BP_GRAD_Z, gradz_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Z, f"dL/dz = {values['grad_z']:.3f}", SNN_COLOR, alpha=gradz_reveal, w=1.6)
        self._equation_near(*self._BP_GRAD_Z, "dL/dz = dL/dy · σ'(z)", gradz_reveal, dy=-0.5)

        self._flow_arrow(self._BP_GRAD_Z, self._BP_GRAD_W, gradw_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_W, f"dL/dw = {values['grad_w']:.2f}", SNN_COLOR, alpha=gradw_reveal, w=1.5)
        self._equation_near(*self._BP_GRAD_W, "dL/dw = dL/dz · x", gradw_reveal, dy=-0.55)

        self._flow_arrow(self._BP_GRAD_W, self._BP_W, update_reveal, ACCENT_COLOR)
        self._fading_text(0.7, 2.0, f"atualizado -> {values['w_updated']:.3f}", ACCENT_COLOR, update_reveal, fontsize=8)
        self._equation_near(*self._BP_W, "w ← w - taxa · dL/dw", update_reveal, dy=-0.55)

        ax = self._get_inset("main", (0.56, 0.1, 0.42, 0.85))
        self._paint_sigmoid(
            ax, z=float(values["z"]), y=float(values["y"]), slope=float(values["slope"]), grad_z=float(values["grad_z"]),
            point_reveal=y_reveal, tangent_reveal=y_reveal, arrow_reveal=gradz_reveal,
            title="Ativação: onde estamos na curva",
        )

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
        w1_real, w2_real = values["w_real"]
        w1q, w2q = values["w_quant"]
        threshold = float(values["threshold"])
        quant1_reveal = float(values["quant1_reveal"])
        quant2_reveal = float(values["quant2_reveal"])
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

        w_label1 = f"Q(w1) = {round(w1q):+d}" if quant1_reveal >= 0.999 else f"w1 = {w1_real:.2f}"
        w_label2 = f"Q(w2) = {round(w2q):+d}" if quant2_reveal >= 0.999 else f"w2 = {w2_real:.2f}"
        self._box(*self._W1, w_label1, BITNET_COLOR if quant1_reveal > 0.5 else NEUTRAL_COLOR, glow=h1)
        self._box(*self._W2, w_label2, BITNET_COLOR if quant2_reveal > 0.5 else NEUTRAL_COLOR, glow=h2)
        self._equation_near(*self._W1, "Q(w)=+1 se w>τ; -1 se w<-τ; 0 c.c.", max(quant1_reveal, quant2_reveal))

        self._flow_arrow(self._W1, self._SUM, arrow1_fill, ACCENT_COLOR if h1 > 0.3 else BITNET_COLOR)
        self._flow_arrow(self._W2, self._SUM, arrow2_fill, ACCENT_COLOR if h2 > 0.3 else BITNET_COLOR)
        self._fading_text(4.9, 4.35, f"produto = {values['product1']:g}", ACCENT_COLOR, p1_reveal, fontsize=9)
        self._fading_text(4.9, 1.95, f"produto = {values['product2']:g}", ACCENT_COLOR, p2_reveal, fontsize=9)

        self._box(*self._SUM, "soma", CONVERGE_COLOR, alpha=max(sum_reveal, 0.35 if sum_reveal else 0.0) or 0.01)
        self._fading_text(*self._SUM, "Σ", CONVERGE_COLOR, 1.0 if sum_reveal < 0.02 else 0.0, fontsize=14, weight="bold")
        self._equation_near(*self._SUM, "y = Σ xᵢ·Q(wᵢ)", sum_reveal)

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
        self._equation_near(*self._LOSS, "L = ½ (y - target)²", loss_reveal)

        ax = self._get_inset("forward_numberline", (0.56, 0.12, 0.42, 0.8))
        self._paint_weight_numberline(ax, w1_real, w2_real, w1q, w2q, threshold, quant1_reveal, quant2_reveal)

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
        self._fading_text(5.1, 5.02, "y usa Q(w) = +1/0/-1", BITNET_COLOR, fwd, fontsize=8)

        for pos, label in zip(self._BWD_POSITIONS, self._BWD_LABELS):
            self._box(*pos, label, SNN_COLOR, alpha=bwd, w=2.0)
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._flow_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]), bwd, SNN_COLOR)
        self._fading_text(5.1, 0.65, "BACKWARD (STE)", SNN_COLOR, bwd, fontsize=12, weight="bold")
        self._fading_text(5.1, 0.98, "dL/dw_real ≈ dL/dy  (dQ/dw trocada por 1)", SNN_COLOR, bwd, fontsize=8)

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
        self._equation_near(*self._Q_POS, "Q(w) = ±1/0 (limiar τ)", q_reveal, dy=-0.95)

        self._box(*self._X_POS, f"x = {2.0:g}", NEUTRAL_COLOR, alpha=x_reveal)
        self._flow_arrow(self._Q_POS, self._Y_POS_G, y_reveal, CONVERGE_COLOR)
        self._flow_arrow(self._X_POS, self._Y_POS_G, y_reveal, NEUTRAL_COLOR)
        self._box(*self._Y_POS_G, f"y = {values['y_value']:g}", CONVERGE_COLOR, alpha=y_reveal)
        self._equation_near(*self._Y_POS_G, "y = x · Q(w)", y_reveal)

        self._flow_arrow(self._Y_POS_G, self._TARGET_POS_G, target_reveal, NEUTRAL_COLOR)
        self._box(*self._TARGET_POS_G, f"target = {values['target_value']:g}", NEUTRAL_COLOR, alpha=target_reveal)

        self._flow_arrow(self._Y_POS_G, self._LOSS_POS_G, loss_reveal, SNN_COLOR)
        self._box(*self._LOSS_POS_G, f"loss = {values['loss_value']:g}", SNN_COLOR, alpha=loss_reveal)
        self._equation_near(*self._LOSS_POS_G, "L = ½ (y - target)²", loss_reveal)

        self._flow_arrow(self._LOSS_POS_G, self._GRAD_POS, grad_reveal, SNN_COLOR)
        self._box(*self._GRAD_POS, f"dL/dw ~= {values['grad_value']:g}", SNN_COLOR, alpha=grad_reveal)
        self._equation_near(*self._GRAD_POS, "dL/dw ≈ dL/dy  (STE)", grad_reveal, dy=0.62)

        self._flow_arrow(self._GRAD_POS, self._W_POS, update_reveal, ACCENT_COLOR, )
        self._fading_text(3.5, 0.15, "STE: gradiente atravessa Q(w) como identidade", ACCENT_COLOR, ste_reveal, fontsize=8)
        self._fading_text(1.2, 4.1, "atualizado", ACCENT_COLOR, update_reveal, fontsize=9)
        self._equation_near(*self._W_POS, "w ← w - taxa · dL/dw", update_reveal)

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
