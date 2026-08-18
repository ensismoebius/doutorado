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

import textwrap
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

    def _fading_text(
        self, x: float, y: float, text: str, color: str, alpha: float,
        fontsize: float = 10, weight: str = "normal", ha: str = "center",
    ) -> None:
        if alpha <= 0.02:
            return
        self._ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, alpha=alpha, fontweight=weight)

    def _equation_near(
        self, x: float, y: float, text: str, alpha: float,
        box_h: float = 0.85, side: str = "below", fontsize: float = 10,
    ) -> None:
        """A formula placed against the box whose value it explains.

        Offset is computed from the box height plus the text's own rendered
        height (derived from the axes' data span and the figure's physical
        size), so a larger font never lands on top of the box it annotates --
        the old hard-coded dy values were only correct for 7.5pt text and
        slid the formula onto the box edge the moment the font grew. A white
        chip is drawn behind the text (text zorder sits above the arrows' 2)
        so a formula reads cleanly even where a diagram arrow passes under
        it. Kept visually distinct (italic, grey) from the box's own label so
        it always reads as "this is the rule", not another value.
        """
        if alpha <= 0.02:
            return
        pos = self._ax.get_position()
        ax_h_in = pos.height * self._figure.get_figheight()
        ymin, ymax = self._ax.get_ylim()
        text_half = fontsize * 1.6 / 72.0 * (ymax - ymin) / ax_h_in / 2.0
        offset = box_h / 2.0 + text_half + 0.16
        dy = -offset if side == "below" else offset
        self._ax.text(
            x, y + dy, text, ha="center", va="center", fontsize=fontsize, color=NEUTRAL_COLOR,
            alpha=alpha, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.95 * alpha),
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
                ax.text(z, y + 0.08, f"y = σ({z:.2f}) = {y:.2f}", ha="center", fontsize=9, color=CONVERGE_COLOR, alpha=point_reveal)

        if tangent_reveal > 0.02:
            half = 2.0
            z_tan = np.array([z - half, z + half])
            y_tan = y + slope * (z_tan - z)
            ax.plot(z_tan, y_tan, color=ACCENT_COLOR, linewidth=1.3 if compact else 1.5, linestyle="--", alpha=tangent_reveal)
            if not compact:
                ax.text(
                    z_tan[0], y_tan[0], f"σ'(z) = {slope:.2f}", ha="right", va="top",
                    fontsize=8.5, color=ACCENT_COLOR, alpha=tangent_reveal,
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
                    "descida do gradiente", ha="center", fontsize=8.5, color=SNN_COLOR, alpha=arrow_reveal,
                )

        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-0.15, 1.15)
        # set_xlabel/set_ylabel/set_title/tick_params each force matplotlib
        # to recompute tick/label layout (Axis._update_label_position) on
        # the *next* draw regardless of whether the text actually changed —
        # measured as a meaningful chunk of this inset's per-frame redraw
        # cost. fast_clear() never touches these (only lines/patches/texts/
        # collections/images), so they stay correct across frames without
        # being re-set; only re-set when the value genuinely changed (e.g.
        # switching which neuron's panel this cached Axes now shows).
        if compact:
            if ax.get_title() != title:
                ax.set_title(title, fontsize=8.5)
            if ax.get_xticks().size or ax.get_yticks().size:
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            if ax.get_xlabel() != "z":
                ax.set_xlabel("z", fontsize=8)
            if ax.get_ylabel() != "σ(z)":
                ax.set_ylabel("σ(z)", fontsize=8)
            if ax.get_title() != title:
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
        ax.text(threshold, -0.42, f"τ = {threshold:.2f}", ha="left", fontsize=9, color=NEUTRAL_COLOR, style="italic")
        ax.text(-threshold, -0.42, f"-τ = {-threshold:.2f}", ha="right", fontsize=9, color=NEUTRAL_COLOR, style="italic")

        def draw(w: float, wq: int, reveal: float, row_y: float, label: str) -> None:
            reveal = max(0.0, min(1.0, reveal))
            display = w + (wq - w) * reveal
            color = ACCENT_COLOR if reveal >= 0.999 else BITNET_COLOR
            ax.plot([display], [row_y], marker="o", markersize=12, color=color, zorder=4)
            text = f"Q({label}) = {round(wq):+d}" if reveal >= 0.999 else f"{label} = {w:.2f}"
            ax.text(display, row_y + 0.14, text, ha="center", fontsize=9, color=color)

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
    _MLP_NAMES = ["L1-A", "L1-B", "L2-C", "L2-D", "O"]
    _MLP_POS = {"L1-A": _MLP_L1[0], "L1-B": _MLP_L1[1], "L2-C": _MLP_L2[0], "L2-D": _MLP_L2[1], "O": _MLP_O}
    _MLP_INPUT_POS = {
        "L1-A": _MLP_X, "L1-B": _MLP_X,
        "L2-C": _MLP_L1, "L2-D": _MLP_L1,
        "O": _MLP_L2,
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
        "O": (0.86, 0.32, 0.13, 0.38),
    }

    def _render_mlp_network(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(-0.1, 8.6), ylim=(-3.0, 6.6))
        x, target = values["x"], float(values["target"])
        w1, w2, w3 = values["w1"], values["w2"], values["w3"]
        z1, y1, z2, y2, zO, yO = values["z1"], values["y1"], values["z2"], values["y2"], float(values["zO"]), float(values["yO"])
        gz1, gz2, gzO = values["grad_z1"], values["grad_z2"], float(values["grad_zO"])
        fwd = {
            "L1-A": float(values["fwd_l1a"]), "L1-B": float(values["fwd_l1b"]),
            "L2-C": float(values["fwd_l2c"]), "L2-D": float(values["fwd_l2d"]), "O": float(values["fwd_o"]),
        }
        bwd = {
            "L1-A": float(values["bwd_l1a"]), "L1-B": float(values["bwd_l1b"]),
            "L2-C": float(values["bwd_l2c"]), "L2-D": float(values["bwd_l2d"]), "O": float(values["bwd_o"]),
        }
        y_val = {"L1-A": float(y1[0]), "L1-B": float(y1[1]), "L2-C": float(y2[0]), "L2-D": float(y2[1]), "O": yO}
        gz_val = {"L1-A": float(gz1[0]), "L1-B": float(gz1[1]), "L2-C": float(gz2[0]), "L2-D": float(gz2[1]), "O": gzO}
        w_row = {"L1-A": w1[0], "L1-B": w1[1], "L2-C": w2[0], "L2-D": w2[1], "O": w3[0]}
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
                text += f"\n∂L/∂z={gz_val[name]:.3f}"
            self._box(*pos, text, CONVERGE_COLOR if bwd[name] > 0.02 else BITNET_COLOR, alpha=r, w=1.4, h=0.75, glow=glow, fontsize=8)
            if glow:
                self._equation_near(*pos, "y = σ(Σ w·x)", r, box_h=0.75)

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
                0.1, -1.1, detail, ha="left", va="top", fontsize=8.5, color="black",
                family="monospace", linespacing=1.6,
            )

        if update_reveal > 0.02:
            self._fading_text(4.2, -2.3, "todos os pesos atualizados: w ← w - taxa · ∂L/∂w", ACCENT_COLOR, update_reveal, fontsize=9)

        z_val = {"L1-A": float(z1[0]), "L1-B": float(z1[1]), "L2-C": float(z2[0]), "L2-D": float(z2[1]), "O": zO}
        slope_val = {name: float(sigmoid_derivative(z_val[name])) for name in self._MLP_NAMES}
        for name in self._MLP_NAMES:
            ax = self._get_inset(name, self._MLP_INSET_RECTS[name])
            self._paint_sigmoid(
                ax, z=z_val[name], y=y_val[name], slope=slope_val[name], grad_z=gz_val[name],
                point_reveal=fwd[name], tangent_reveal=fwd[name], arrow_reveal=bwd[name],
                title=name, compact=True,
            )

    # ================================================================
    # backprop/demos/matrix_algebra.py — the same network drawn twice at
    # once: a graph on the left, real vectors/matrices on the right. The
    # whole point is that the two are linked cell-by-edge, so a single
    # highlight field (hl_w1[i, j]) glows BOTH the matrix cell and the graph
    # edge x_j -> H_i. Everything is driven by per-cell reveal arrays, which
    # core/math_utils.tween_values interpolates element-wise -- that is what
    # makes a matrix fill in one entry at a time instead of appearing whole.
    #
    # Each node carries its own values INSIDE its box (name / z / y, each
    # with its own alpha). They used to be printed underneath, which put
    # them in the path of the O -> L arrow and forced the boxes to stay
    # small to leave room; in-box means bigger boxes, bigger type, and no
    # arrow striking through a number.
    # ================================================================
    #: Column spacing has to exceed the two nodes' radii plus a visible
    #: stretch of arrow, or the trimmed edge between adjacent columns comes
    #: out zero-length and the connection simply disappears.
    _MA_NODES = {
        "x1": (0.90, 8.15), "x2": (0.90, 5.75),
        "h1": (3.25, 8.15), "h2": (3.25, 5.75),
        "o": (5.60, 6.95),
    }
    _MA_LOSS = (5.60, 4.05)
    _MA_IN_W, _MA_IN_H = 1.05, 0.95     # 2 lines: name + value
    _MA_MID_W, _MA_MID_H = 1.32, 1.35   # 3 lines: name + z + y
    _MA_LOSS_W, _MA_LOSS_H = 1.7, 0.95  # 2 lines: alvo + L
    #: node "radius" used to trim edges to the box outline (boxstyle pad
    #: 0.25 inflates every box beyond its nominal width, so this is
    #: half-width + that pad); without trimming, the semi-transparent boxes
    #: let the arrows show through and cross the numbers inside.
    _MA_RADIUS = {"x1": 0.775, "x2": 0.775, "h1": 0.91, "h2": 0.91, "o": 0.91}

    #: The board is a row of items ([matriz] · [entrada] = [saída]) laid out
    #: by _board_row, not at fixed columns: the matrices change shape between
    #: phases (2x2, 1x2, 2x1, 1x1), and fixed columns tuned for the widest
    #: case leave the operators jammed against the brackets of the narrow
    #: ones. Widths are measured, then the row is centred.
    _MA_ROW1_Y = 7.7    # the multiplication / main operation
    _MA_ACCUM_Y = 5.72  # running partial sum, clear of row 1's captions
    _MA_ROW2_Y = 4.18   # the activation (its own row: it is its own step)
    _MA_BOARD_CX = 11.6
    _MA_BOARD_GAP = 0.24
    _MA_OP_W, _MA_ARROW_W = 0.5, 0.95
    _MA_CELL_W, _MA_CELL_H, _MA_CELL_FS = 1.22, 0.84, 10.0
    _MA_CHAIN_X = (7.70, 9.55, 11.40, 13.25, 15.10)
    _MA_CHAIN_W = 1.2

    def _mx_grid(
        self, cx: float, cy: float, arr, reveal, hl, color: str,
        row_labels: tuple[str, ...] = (), col_labels: tuple[str, ...] = (),
        decimals: int = 2, caption: str = "",
    ) -> None:
        """A bracketed numeric grid whose cells reveal/glow independently.

        ``reveal`` and ``hl`` are same-shape arrays of 0..1: reveal fades a
        cell's number in (0 leaves a faint placeholder dash, so the grid's
        *shape* is legible before any value exists), hl paints the
        highlight chip behind it. Brackets are always drawn -- the empty
        matrix is part of the picture from frame one.
        """
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        reveal = np.atleast_2d(np.asarray(reveal, dtype=float)).reshape(arr.shape)
        hl = np.atleast_2d(np.asarray(hl, dtype=float)).reshape(arr.shape)
        nrows, ncols = arr.shape
        total_w, total_h = ncols * self._MA_CELL_W, nrows * self._MA_CELL_H
        x0, x1 = cx - total_w / 2, cx + total_w / 2
        y0, y1 = cy - total_h / 2, cy + total_h / 2

        for bx, tick in ((x0 - 0.11, 0.2), (x1 + 0.11, -0.2)):
            self._ax.plot([bx, bx], [y0, y1], color=color, linewidth=2.2, alpha=0.9, solid_capstyle="butt")
            self._ax.plot([bx, bx + tick], [y1, y1], color=color, linewidth=2.2, alpha=0.9, solid_capstyle="butt")
            self._ax.plot([bx, bx + tick], [y0, y0], color=color, linewidth=2.2, alpha=0.9, solid_capstyle="butt")

        for i in range(nrows):
            for j in range(ncols):
                ccx = x0 + (j + 0.5) * self._MA_CELL_W
                ccy = y1 - (i + 0.5) * self._MA_CELL_H
                if hl[i, j] > 0.02:
                    self._ax.add_patch(FancyBboxPatch(
                        (ccx - self._MA_CELL_W / 2 + 0.06, ccy - self._MA_CELL_H / 2 + 0.05),
                        self._MA_CELL_W - 0.12, self._MA_CELL_H - 0.1,
                        facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.55 * float(hl[i, j]),
                        boxstyle="round,pad=0.04", zorder=1,
                    ))
                rv = float(reveal[i, j])
                if rv > 0.02:
                    self._ax.text(
                        ccx, ccy, f"{arr[i, j]:+.{decimals}f}", ha="center", va="center",
                        fontsize=self._MA_CELL_FS, family="monospace", color="black",
                        alpha=rv, zorder=3,
                    )
                if rv < 0.98:
                    self._ax.text(
                        ccx, ccy, "·", ha="center", va="center", fontsize=11,
                        color=NEUTRAL_COLOR, alpha=_SKELETON_ALPHA * (1.0 - rv), zorder=2,
                    )

        for j, label in enumerate(col_labels):
            self._ax.text(
                x0 + (j + 0.5) * self._MA_CELL_W, y1 + 0.22, label, ha="center", va="bottom",
                fontsize=9, color=NEUTRAL_COLOR, style="italic",
            )
        for i, label in enumerate(row_labels):
            self._ax.text(
                x0 - 0.42, y1 - (i + 0.5) * self._MA_CELL_H, label, ha="right", va="center",
                fontsize=9, color=NEUTRAL_COLOR, style="italic",
            )
        if caption:
            self._ax.text(
                cx, y0 - 0.32, caption, ha="center", va="top", fontsize=9.5,
                color=color, style="italic",
            )

    def _node_box(
        self, pos: tuple[float, float], w: float, h: float, color: str,
        lines: tuple[tuple[str, float], ...], box_alpha: float, glow: float = 0.0,
        fontsize: float = 9.5,
    ) -> None:
        """A graph node whose text lines each fade in on their own.

        ``lines`` is ``((text, alpha), ...)`` top to bottom. Separate alphas
        are what let the animation show "z is computed but y is not yet" --
        a single box label could only appear all at once.
        """
        x, y = pos
        if glow > 0.02:
            self._ax.add_patch(FancyBboxPatch(
                (x - w / 2 - 0.13, y - h / 2 - 0.13), w + 0.26, h + 0.26,
                facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.5 * glow,
                boxstyle="round,pad=0.25",
            ))
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor=color, alpha=_FILL_ALPHA * box_alpha,
            edgecolor=color, **_BOX_STYLE,
        )
        self._ax.add_patch(patch)
        n = len(lines)
        step = h / max(1, n)
        for k, (text, alpha) in enumerate(lines):
            if alpha <= 0.02:
                continue
            ly = y + h / 2 - step * (k + 0.5)
            self._ax.text(
                x, ly, text, ha="center", va="center", fontsize=fontsize if k == 0 else fontsize - 1.2,
                color="black", alpha=alpha, zorder=4,
                fontweight="bold" if k == 0 else "normal",
            )

    def _ma_edge(self, src: str, dest: str, glow: float) -> None:
        """One graph edge, trimmed to both node outlines."""
        n = self._MA_NODES
        p0, p1 = np.array(n[src], dtype=float), np.array(n[dest], dtype=float)
        direction = p1 - p0
        length = float(np.hypot(*direction))
        unit = direction / length
        a = p0 + unit * self._MA_RADIUS[src]
        b = p1 - unit * self._MA_RADIUS[dest]
        self._skeleton_arrow(tuple(a), tuple(b))
        self._flow_arrow(tuple(a), tuple(b), glow, ACCENT_COLOR)

    def _board_row(self, items: list[dict], y: float) -> None:
        """Lay a row of grids/operators out left-to-right, centred on the board.

        ``items`` are dicts tagged by ``t``: ``"grid"`` (keys arr/rv/hl/color
        plus optional rows/cols/dec/cap), ``"op"`` (a glyph between two
        grids) or ``"arrow"`` (a flow arrow with a glyph above it, used for
        the σ / ½()² step). Each item's width is measured from its actual
        shape first, so operators always land in the middle of the gap
        whatever the matrices' dimensions are.
        """
        widths = []
        for item in items:
            if item["t"] == "op":
                widths.append(self._MA_OP_W)
            elif item["t"] == "arrow":
                widths.append(self._MA_ARROW_W)
            elif item["t"] == "grid":
                ncols = np.atleast_2d(np.asarray(item["arr"], dtype=float)).shape[1]
                widths.append(ncols * self._MA_CELL_W + 0.22)
            else:
                raise ValueError(f"_board_row: unknown item type {item['t']!r}")
        total = sum(widths) + self._MA_BOARD_GAP * (len(items) - 1)
        x = self._MA_BOARD_CX - total / 2.0
        for item, width in zip(items, widths):
            cx = x + width / 2.0
            if item["t"] == "grid":
                self._mx_grid(
                    cx, y, item["arr"], item["rv"], item["hl"], item["color"],
                    row_labels=item.get("rows", ()), col_labels=item.get("cols", ()),
                    decimals=item.get("dec", 2), caption=item.get("cap", ""),
                )
            elif item["t"] == "op":
                self._fading_text(
                    cx, y, item["glyph"], NEUTRAL_COLOR, item.get("alpha", 1.0),
                    fontsize=item.get("size", 24 if item["glyph"] == "·" else 19), weight="bold",
                )
            else:
                reveal = item.get("alpha", 1.0)
                self._skeleton_arrow((x, y), (x + width, y))
                self._flow_arrow((x, y), (x + width, y), reveal, item["color"])
                self._fading_text(
                    cx, y + 0.52, item["glyph"], item["color"], reveal,
                    fontsize=item.get("size", 15), weight="bold",
                )
            x += width + self._MA_BOARD_GAP

    def _render_matrix_algebra(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(-0.1, 16.1), ylim=(0.4, 10.0))
        x, w1, w2 = values["x"], values["w1"], values["w2"]
        z1, y1 = values["z1"], values["y1"]
        z2, y2 = float(values["z2"]), float(values["y2"])
        sp1, sp2 = values["sp1"], float(values["sp2"])
        target, diff, loss = float(values["target"]), float(values["diff"]), float(values["loss"])
        gy2, gz2 = float(values["gy2"]), float(values["gz2"])
        gw2, gy1, gz1, gw1 = values["gw2"], values["gy1"], values["gz1"], values["gw1"]
        rv_graph = float(values["rv_graph"])
        rv_x, rv_w1, rv_w2 = values["rv_x"], values["rv_w1"], values["rv_w2"]
        rv_z1, rv_y1 = values["rv_z1"], values["rv_y1"]
        rv_z2, rv_y2 = float(values["rv_z2"]), float(values["rv_y2"])
        rv_target, rv_diff, rv_loss = float(values["rv_target"]), float(values["rv_diff"]), float(values["rv_loss"])
        rv_sp1, rv_sp2 = values["rv_sp1"], float(values["rv_sp2"])
        hl_w1, hl_w2 = values["hl_w1"], values["hl_w2"]
        hl_x, hl_y1, hl_out = values["hl_x"], values["hl_y1"], float(values["hl_out"])
        focus = str(values["focus"])
        n = self._MA_NODES

        # -- the graph, left. hl_w1/hl_w2 are the SAME arrays that glow the
        # matrix cells, which is the mapping this demo is about.
        for i, dest in enumerate(("h1", "h2")):
            for j, src in enumerate(("x1", "x2")):
                self._ma_edge(src, dest, float(hl_w1[i, j]))
        for j, src in enumerate(("h1", "h2")):
            self._ma_edge(src, "o", float(hl_w2[0, j]))
        o_bottom = (n["o"][0], n["o"][1] - self._MA_MID_H / 2 - 0.25)
        loss_top = (self._MA_LOSS[0], self._MA_LOSS[1] + self._MA_LOSS_H / 2 + 0.25)
        self._skeleton_arrow(o_bottom, loss_top)
        self._flow_arrow(o_bottom, loss_top, rv_loss, SNN_COLOR)

        self._node_box(
            n["x1"], self._MA_IN_W, self._MA_IN_H, NEUTRAL_COLOR,
            (("x1", rv_graph), (f"{float(x[0]):+.2f}", float(rv_x[0]))),
            rv_graph, glow=float(hl_x[0]),
        )
        self._node_box(
            n["x2"], self._MA_IN_W, self._MA_IN_H, NEUTRAL_COLOR,
            (("x2", rv_graph), (f"{float(x[1]):+.2f}", float(rv_x[1]))),
            rv_graph, glow=float(hl_x[1]),
        )
        for key, label, idx in (("h1", "H1", 0), ("h2", "H2", 1)):
            self._node_box(
                n[key], self._MA_MID_W, self._MA_MID_H, BITNET_COLOR,
                (
                    (label, rv_graph),
                    (f"z={float(z1[idx]):+.3f}", float(rv_z1[idx])),
                    (f"y={float(y1[idx]):.3f}", float(rv_y1[idx])),
                ),
                rv_graph, glow=float(hl_y1[idx]),
            )
        self._node_box(
            n["o"], self._MA_MID_W, self._MA_MID_H, CONVERGE_COLOR,
            (("O", rv_graph), (f"z={z2:+.3f}", rv_z2), (f"y={y2:.3f}", rv_y2)),
            rv_graph, glow=hl_out,
        )
        self._node_box(
            self._MA_LOSS, self._MA_LOSS_W, self._MA_LOSS_H, SNN_COLOR,
            ((f"alvo = {target:g}", rv_target), (f"L = {loss:.4f}", rv_loss)),
            max(rv_target, rv_loss),
        )
        self._fading_text(3.25, 9.6, "a rede como grafo", NEUTRAL_COLOR, 1.0, fontsize=10.5, weight="bold")

        # -- the algebra board, right. Row 1 is the multiplication; row 2 is
        # the activation, which gets its own row because it is its own step.
        self._fading_text(
            self._MA_BOARD_CX, 9.6, str(values["board_title"]), BITNET_COLOR, 1.0,
            fontsize=11.5, weight="bold",
        )
        row1, row2 = self._MA_ROW1_Y, self._MA_ROW2_Y
        zero = np.zeros((1, 1))

        def grid(arr, rv, hl, color, **kw) -> dict:
            return {"t": "grid", "arr": arr, "rv": rv, "hl": hl, "color": color, **kw}

        def scalar(value, rv, color, hl=None, **kw) -> dict:
            return grid(
                np.array([[value]]), np.array([[rv]]),
                zero if hl is None else np.array([[hl]]), color, dec=4, **kw
            )

        if focus == "chain":
            names, vals = values["chain_names"], values["chain_values"]
            rv_chain = values["rv_chain"]
            for k, cx in enumerate(self._MA_CHAIN_X):
                self._node_box(
                    (cx, row1), self._MA_CHAIN_W, 1.15, BITNET_COLOR,
                    ((names[k], float(rv_chain[k])), (f"{vals[k]:+.4f}", float(rv_chain[k]))),
                    float(rv_chain[k]), fontsize=9.5,
                )
                if k:
                    self._fading_text(
                        (cx + self._MA_CHAIN_X[k - 1]) / 2, row1, "·", NEUTRAL_COLOR,
                        float(rv_chain[k]), fontsize=26, weight="bold",
                    )
            self._node_box(
                (self._MA_BOARD_CX, self._MA_ACCUM_Y), 4.8, 0.8, ACCENT_COLOR,
                ((f"produto = {float(values['chain_product']):+.5f}", 1.0),),
                float(values["rv_chain_product"]), fontsize=11,
            )
            self._node_box(
                (self._MA_BOARD_CX, row2), 5.6, 0.8, CONVERGE_COLOR,
                ((f"= grad_W1[H1,x1] = {float(np.asarray(gw1)[0, 0]):+.5f}  ✓", 1.0),),
                float(values["rv_check"]), fontsize=11,
            )
        elif focus in ("l1", "l2"):
            first = focus == "l1"
            out_vec = z1 if first else np.array([z2])
            act_vec = y1 if first else np.array([y2])
            rv_out = rv_z1 if first else np.array([rv_z2])
            rv_act = rv_y1 if first else np.array([rv_y2])
            self._board_row([
                grid(
                    w1 if first else w2, rv_w1 if first else rv_w2, hl_w1 if first else hl_w2,
                    BITNET_COLOR, rows=("H1", "H2") if first else ("O",),
                    cols=("x1", "x2") if first else ("H1", "H2"),
                    cap="W1 (2×2)" if first else "W2 (1×2)",
                ),
                {"t": "op", "glyph": "·"},
                grid(
                    (x if first else y1).reshape(-1, 1),
                    np.asarray(rv_x if first else rv_y1).reshape(-1, 1),
                    np.asarray(hl_x if first else hl_y1).reshape(-1, 1),
                    NEUTRAL_COLOR, cap="x" if first else "y (camada 1)",
                ),
                {"t": "op", "glyph": "="},
                grid(
                    out_vec.reshape(-1, 1), np.asarray(rv_out).reshape(-1, 1),
                    np.zeros((len(out_vec), 1)), CONVERGE_COLOR, dec=3, cap="z",
                ),
            ], row1)
            # the activation, on its own row -- one step of the walkthrough
            act_reveal = float(np.max(np.asarray(rv_act)))
            self._board_row([
                grid(
                    out_vec.reshape(-1, 1), np.asarray(rv_out).reshape(-1, 1),
                    np.zeros((len(out_vec), 1)), CONVERGE_COLOR, dec=3, cap="z",
                ),
                {"t": "arrow", "glyph": "σ", "color": ACCENT_COLOR, "alpha": act_reveal},
                grid(
                    act_vec.reshape(-1, 1), np.asarray(rv_act).reshape(-1, 1),
                    np.zeros((len(act_vec), 1)), CONVERGE_COLOR, dec=3, cap="y = σ(z)",
                ),
            ], row2)
        elif focus == "loss":
            self._board_row([
                scalar(y2, rv_y2, CONVERGE_COLOR, hl=hl_out, cap="y_O (rede)"),
                {"t": "op", "glyph": "−"},
                grid(np.array([[target]]), np.array([[rv_target]]), zero, NEUTRAL_COLOR, dec=2, cap="alvo"),
                {"t": "op", "glyph": "="},
                scalar(diff, rv_diff, SNN_COLOR, cap="diferença"),
            ], row1)
            self._board_row([
                scalar(diff, rv_diff, SNN_COLOR, cap="diferença"),
                {"t": "arrow", "glyph": "½( )²", "color": SNN_COLOR, "alpha": rv_loss, "size": 12},
                scalar(loss, rv_loss, SNN_COLOR, cap="L"),
            ], row2)
        elif focus == "gz2":
            self._board_row([
                scalar(gy2, float(values["rv_gy2"]), SNN_COLOR, cap="∂L/∂y_O"),
                {"t": "op", "glyph": "·"},
                scalar(sp2, rv_sp2, ACCENT_COLOR, hl=hl_out, cap="σ'(z_O)"),
                {"t": "op", "glyph": "="},
                scalar(gz2, float(values["rv_gz2"]), SNN_COLOR, cap="∂L/∂z_O"),
            ], row1)
        elif focus == "gw2":
            self._board_row([
                scalar(gz2, 1.0, SNN_COLOR, cap="∂L/∂z_O"),
                {"t": "op", "glyph": "⊗"},
                grid(y1.reshape(1, -1), np.ones((1, 2)), np.asarray(hl_y1).reshape(1, -1), CONVERGE_COLOR, dec=3, cap="y (linha)"),
                {"t": "op", "glyph": "="},
                grid(gw2, values["rv_gw2"], np.zeros((1, 2)), SNN_COLOR, rows=("O",), cols=("H1", "H2"), dec=4, cap="grad_W2 (1×2)"),
            ], row1)
        elif focus == "w2t":
            self._board_row([
                grid(w2.T, np.asarray(rv_w2).T, np.asarray(hl_w2).T, BITNET_COLOR, rows=("H1", "H2"), cols=("O",), cap="W2ᵀ (2×1)"),
                {"t": "op", "glyph": "·"},
                scalar(gz2, 1.0, SNN_COLOR, cap="∂L/∂z_O"),
                {"t": "op", "glyph": "="},
                grid(gy1.reshape(-1, 1), np.asarray(values["rv_gy1"]).reshape(-1, 1), np.zeros((2, 1)), SNN_COLOR, dec=4, cap="∂L/∂y"),
            ], row1)
        elif focus == "gz1":
            self._board_row([
                grid(gy1.reshape(-1, 1), np.ones((2, 1)), np.zeros((2, 1)), SNN_COLOR, rows=("H1", "H2"), dec=4, cap="∂L/∂y"),
                {"t": "op", "glyph": "⊙"},
                grid(
                    np.asarray(sp1).reshape(-1, 1), np.asarray(rv_sp1).reshape(-1, 1),
                    np.asarray(hl_y1).reshape(-1, 1), ACCENT_COLOR, dec=4, cap="σ'(z)",
                ),
                {"t": "op", "glyph": "="},
                grid(gz1.reshape(-1, 1), np.asarray(values["rv_gz1"]).reshape(-1, 1), np.zeros((2, 1)), SNN_COLOR, dec=4, cap="∂L/∂z"),
            ], row1)
        elif focus == "gw1":
            self._board_row([
                grid(
                    gz1.reshape(-1, 1), np.ones((2, 1)), np.asarray(hl_y1).reshape(-1, 1),
                    SNN_COLOR, rows=("H1", "H2"), dec=4, cap="∂L/∂z (coluna)",
                ),
                {"t": "op", "glyph": "⊗"},
                grid(x.reshape(1, -1), np.ones((1, 2)), np.asarray(hl_x).reshape(1, -1), NEUTRAL_COLOR, cap="x (linha)"),
                {"t": "op", "glyph": "="},
                grid(gw1, values["rv_gw1"], np.zeros((2, 2)), SNN_COLOR, rows=("H1", "H2"), cols=("x1", "x2"), dec=4, cap="grad_W1 (2×2)"),
            ], row1)
        else:
            raise ValueError(f"matrix_algebra: unknown focus {focus!r}")

        # -- the running partial sum: a real interpolated number, so it
        # visibly counts up/down while the transition plays instead of
        # jumping to the answer.
        rv_accum = float(values["rv_accum"])
        if rv_accum > 0.02 and focus != "chain":
            self._node_box(
                (self._MA_BOARD_CX, self._MA_ACCUM_Y), 4.6, 0.8, ACCENT_COLOR,
                ((f"soma parcial = {float(values['accum']):+.4f}", 1.0),), rv_accum, fontsize=11,
            )

        work = str(values["work_text"])
        if work:
            self._ax.text(
                0.0, 2.95, work, ha="left", va="top", fontsize=9.2, color="black",
                family="monospace", linespacing=1.55,
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
        # Widened defensively past the box/equation-text extents below
        # (tests/test_widgets_no_clipping.py found no actual overflow, but
        # _BP_Y's box came within 0.05 units of xlim and _BP_GRAD_Z's
        # equation text already sits past the old ylim, only absorbed by
        # this axes' own figure-position padding) -- extra headroom here
        # costs nothing and removes the fragility.
        self._reset_axes(xlim=(-0.5, 6.9), ylim=(-0.7, 7.0))
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
        self._box(*self._BP_Z, f"z = {values['z']:g}", BITNET_COLOR, alpha=z_reveal, w=1.5, glow=float(values.get("z_glow", 0.0)))
        self._equation_near(*self._BP_Z, "z = w · x", z_reveal)

        self._flow_arrow(self._BP_Z, self._BP_Y, y_reveal, CONVERGE_COLOR)
        self._box(*self._BP_Y, f"y = σ(z)\n= {values['y']:.3f}", CONVERGE_COLOR, alpha=y_reveal, w=1.9, glow=float(values.get("y_glow", 0.0)))
        self._equation_near(*self._BP_Y, "y = σ(z) = 1/(1+e⁻ᶻ)", y_reveal)

        self._flow_arrow(self._BP_Y, self._BP_TARGET, target_reveal, NEUTRAL_COLOR)
        self._box(*self._BP_TARGET, f"target = {values['target']:g}", NEUTRAL_COLOR, alpha=target_reveal, w=1.5, glow=float(values.get("target_glow", 0.0)))
        self._fading_text(
            (self._BP_Y[0] + self._BP_TARGET[0]) / 2 - 1.05, (self._BP_Y[1] + self._BP_TARGET[1]) / 2,
            f"diferença = {values['diff']:.3f}", ACCENT_COLOR, diff_reveal, fontsize=8,
        )

        self._flow_arrow(self._BP_Y, self._BP_LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._BP_LOSS, f"loss = {values['loss']:.3f}", SNN_COLOR, alpha=loss_reveal, w=1.5, glow=float(values.get("loss_glow", 0.0)))
        self._equation_near(*self._BP_LOSS, "L = ½ (y - target)²", loss_reveal)

        self._flow_arrow(self._BP_LOSS, self._BP_GRAD_Y, grady_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Y, f"∂L/∂y = {values['grad_y']:.2f}", SNN_COLOR, alpha=grady_reveal, w=1.5, glow=float(values.get("grady_glow", 0.0)))
        self._equation_near(*self._BP_GRAD_Y, "∂L/∂y = y - target", grady_reveal)

        self._flow_arrow(self._BP_GRAD_Y, self._BP_GRAD_Z, gradz_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Z, f"∂L/∂z = {values['grad_z']:.3f}", SNN_COLOR, alpha=gradz_reveal, w=1.6, glow=float(values.get("gradz_glow", 0.0)))
        self._equation_near(*self._BP_GRAD_Z, "∂L/∂z = ∂L/∂y · σ'(z)", gradz_reveal)

        self._flow_arrow(self._BP_GRAD_Z, self._BP_GRAD_W, gradw_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_W, f"∂L/∂w = {values['grad_w']:.2f}", SNN_COLOR, alpha=gradw_reveal, w=1.5, glow=float(values.get("gradw_glow", 0.0)))
        self._equation_near(*self._BP_GRAD_W, "∂L/∂w = ∂L/∂z · x", gradw_reveal)

        self._flow_arrow(self._BP_GRAD_W, self._BP_W, update_reveal, ACCENT_COLOR)
        self._fading_text(0.7, 2.0, f"atualizado -> {values['w_updated']:.3f}", ACCENT_COLOR, update_reveal, fontsize=8)
        self._equation_near(*self._BP_W, "w ← w - taxa · ∂L/∂w", update_reveal)

        ax = self._get_inset("main", (0.56, 0.1, 0.42, 0.85))
        # point_reveal/arrow_reveal are their own fields, deliberately
        # decoupled from y_reveal/gradz_reveal: those block-diagram reveals
        # reset every repeat cycle (see traditional_gd.py), but the point
        # sliding along this curve must stay continuous across cycles, or
        # the one thing meant to visibly show convergence would itself
        # blink out and back every time.
        inset_point_reveal = float(values.get("point_reveal", y_reveal))
        inset_arrow_reveal = float(values.get("arrow_reveal", gradz_reveal))
        self._paint_sigmoid(
            ax, z=float(values["z"]), y=float(values["y"]), slope=float(values["slope"]), grad_z=float(values["grad_z"]),
            point_reveal=inset_point_reveal, tangent_reveal=inset_point_reveal, arrow_reveal=inset_arrow_reveal,
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
        self._equation_near(*self._W1, "Q(w) ∈ {+1, 0, -1}", max(quant1_reveal, quant2_reveal))

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
        # The one worked example every box references (see backward.py) --
        # carried through every frame so the diagram shows the same numbers
        # as the explanation text. Defaults keep old checkpoints renderable.
        w = float(values.get("w", 0.65))
        wq = int(values.get("w_quant", 1))
        x = float(values.get("x", 2.0))
        y = float(values.get("y", 2.0))
        loss = float(values.get("loss", 2.0))
        upstream = float(values.get("upstream_grad", -2.0))
        dq_real = float(values.get("dq_dw_real", 0.0))
        dq_ste = float(values.get("dq_dw_ste", 1.0))
        dl_real = float(values.get("dl_dw_real", 0.0))
        dl_ste = float(values.get("dl_dw_ste", upstream))
        tau = float(values.get("threshold", 0.5))

        fwd_labels = [
            f"peso real\nw = {w:.2f}",
            "quantização\nQ(w)",
            f"peso ternário\nQ(w) = {wq:+d}",
            f"operação\ny = {x:g}·Q(w) = {y:g}",
        ]
        bwd_labels = [
            f"loss\nL = {loss:g}",
            f"gradiente\n∂L/∂Q(w) = {upstream:g}",
            f"STE\ndQ/dw := {dq_ste:g}",
            f"peso real\n∂L/∂w = {dl_ste:g}",
        ]

        for p in self._FWD_POSITIONS + self._BWD_POSITIONS:
            self._skeleton_box(*p, w=2.0)
        for a, b in zip(self._FWD_POSITIONS, self._FWD_POSITIONS[1:]):
            self._skeleton_arrow((a[0] + 1.0, a[1]), (b[0] - 1.0, b[1]))
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._skeleton_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]))

        for pos, label in zip(self._FWD_POSITIONS, fwd_labels):
            self._box(*pos, label, BITNET_COLOR, alpha=fwd, w=2.0, fontsize=9)
        for a, b in zip(self._FWD_POSITIONS, self._FWD_POSITIONS[1:]):
            self._flow_arrow((a[0] + 1.0, a[1]), (b[0] - 1.0, b[1]), fwd, BITNET_COLOR)
        self._fading_text(5.1, 5.35, "FORWARD", BITNET_COLOR, fwd, fontsize=12, weight="bold")
        self._fading_text(
            5.1, 5.02, f"y = {x:g}·Q(w), com Q({w:.2f}) = {wq:+d} (τ = {tau:g})",
            BITNET_COLOR, fwd, fontsize=8,
        )

        for pos, label in zip(self._BWD_POSITIONS, bwd_labels):
            self._box(*pos, label, SNN_COLOR, alpha=bwd, w=2.0, fontsize=9)
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._flow_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]), bwd, SNN_COLOR)
        self._fading_text(5.1, 0.65, "BACKWARD (STE)", SNN_COLOR, bwd, fontsize=12, weight="bold")
        self._fading_text(
            5.1, 0.98, f"∂L/∂w = {dl_ste:g}  (dQ/dw: {dq_real:g} -> {dq_ste:g})",
            SNN_COLOR, bwd, fontsize=8,
        )

        # the two paths meet at "peso real": a highlighted ring makes the
        # loop visible once both are on screen. The box re-drawn on top
        # uses the forward label (its real value) since that is the one
        # the reader last saw there.
        self._box(*self._FWD_POSITIONS[0], fwd_labels[0], BITNET_COLOR, alpha=fwd, w=2.0, glow=joined, fontsize=9)
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
        self._equation_near(*self._Q_POS, "Q(w) = ±1/0 (limiar τ)", q_reveal)

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
        self._box(*self._GRAD_POS, f"∂L/∂w ~= {values['grad_value']:g}", SNN_COLOR, alpha=grad_reveal)
        self._equation_near(*self._GRAD_POS, "∂L/∂w ≈ ∂L/∂y  (STE)", grad_reveal, side="above")

        self._flow_arrow(self._GRAD_POS, self._W_POS, update_reveal, ACCENT_COLOR, )
        self._fading_text(3.5, 0.15, "STE: gradiente atravessa Q(w) como identidade", ACCENT_COLOR, ste_reveal, fontsize=8)
        self._fading_text(1.2, 4.1, "atualizado", ACCENT_COLOR, update_reveal, fontsize=9)
        self._equation_near(*self._W_POS, "w ← w - taxa · ∂L/∂w", update_reveal)

    # ================================================================
    # comparison — a persistent, growing 3-column table + outputs panel
    #
    # This is the one demo that is *all text*: five rows x three columns of
    # short phrases, plus an outputs row and two footnotes. Its first
    # version placed every cell at a hard-coded data coordinate with a
    # hard-coded fontsize=8 -- which is what "the SNN and ANN comparison
    # fonts are too small" (FIXME.md) reports. Two separate causes:
    #
    #   1. A fixed point size cannot be right for this widget. The Qt
    #      matplotlib canvas keeps its dpi constant and grows the figure's
    #      *inches* as the window grows, so a fontsize=8 cell that looked
    #      acceptable on a 900x400 canvas is 8pt out of a ~1000pt-tall
    #      figure once the app is on a projector -- i.e. it shrinks,
    #      relatively, exactly when legibility matters most. Measured: the
    #      app hands this widget ~1425x1012 px at a normal window size.
    #   2. Even at fontsize 8 the long cells nearly touched their
    #      neighbours, so "just raise the number" would have collided.
    #
    # So the table sizes *itself*. Every coordinate below is an axes
    # fraction (xlim/ylim are 0..1 for this renderer) and the type size is
    # whichever of the two real constraints binds first:
    #
    #   vertical:   every line of text plus its gaps must fit the box
    #   horizontal: a wrapped cell must fit inside one column
    #
    # Whatever height the type could not claim is then spent on the *gaps*
    # (`spread`), so the table fills its box at any canvas size instead of
    # overflowing small ones and hugging the top edge of large ones.
    # ================================================================
    #: Axes box (figure fractions) for this renderer -- see render().
    _CMP_AX_RECT = (0.02, 0.03, 0.96, 0.94)
    #: Width of the row-name column, as a fraction of the axes box. The
    #: three value columns split what is left, equally.
    _CMP_LABEL_FRAC = 0.20
    _CMP_COL_ORDER = ("ANN", "BitNet", "SNN")
    _CMP_COL_COLORS = {"ANN": NEUTRAL_COLOR, "BitNet": BITNET_COLOR, "SNN": SNN_COLOR}
    #: Characters per line inside a value column. 19 is the smallest
    #: budget that still wraps every cell of comparison/ann_bitnet_snn.py
    #: to at most TWO lines ("quantizada conforme" / "arquitetura" is the
    #: binding case -- at 18 it splits into three, which would collide
    #: with the row below). The row-height budget assumes two.
    _CMP_WRAP = 19
    #: Characters per line for the row names ("Operação principal" ->
    #: two lines). Wide enough to keep "Representação" on one line.
    _CMP_LABEL_WRAP = 14
    #: Average glyph advance as a fraction of the font size, for DejaVu
    #: Sans on these mixed-case strings. Measured from a rendered frame
    #: ("normalmente implícito": 21 chars spanning 123pt at fontsize 11 =
    #: 0.53 em/char); 0.56 leaves a margin.
    _CMP_EM_PER_CHAR = 0.56
    #: Baseline-to-baseline distance as a multiple of the font size.
    _CMP_LINE_H = 1.18
    #: The vertical budget, split into what the font fixes and what may be
    #: stretched. Both are in units of "one line of cell text" and both
    #: must be kept in step with the placement cursor in
    #: _render_comparison_pipeline (the overlap test in
    #: tests/test_widgets_no_clipping.py is what catches a drift).
    #:   text: header 1.35 + 5 rows x 2 + outputs 1 + 2 notes x 0.95
    #:   gaps: top .1, header-rule .35, rule-row .5, 4 x .45 between rows,
    #:         row-rule .55, rule-outputs .45, outputs-note .5,
    #:         note-note .3, bottom .15
    _CMP_TEXT_LINES = 1.35 + 10.0 + 1.0 + 1.9
    _CMP_GAP_LINES = 0.1 + 0.35 + 0.5 + 1.8 + 0.55 + 0.45 + 0.5 + 0.3 + 0.15
    #: Legibility clamps on the cell size. The floor is deliberately below
    #: anything usable: it exists only so an absurdly small embedding still
    #: lays out inside its box instead of spilling out of it. The ceiling
    #: stops a very tall canvas from producing comically large type.
    _CMP_FS_MIN = 5.0
    _CMP_FS_MAX = 26.0
    #: Header/footnote sizes, relative to the computed cell size.
    _CMP_HEADER_RATIO = 1.35
    _CMP_NOTE_RATIO = 0.95
    #: Ceiling on the gap stretch. Past ~3x the rows stop reading as one
    #: table and start reading as unrelated lines.
    _CMP_SPREAD_MAX = 3.0

    def _cmp_geometry(self) -> dict[str, float]:
        """Type sizes and the gap stretch that fit this canvas.

        Single source of truth for the layout: change _CMP_WRAP, add a row
        or retune a gap and the type size follows from it automatically.
        """
        fig_w_in, fig_h_in = self._figure.get_size_inches()
        ax_w_pt = fig_w_in * self._CMP_AX_RECT[2] * 72.0
        ax_h_pt = fig_h_in * self._CMP_AX_RECT[3] * 72.0
        col_frac = (1.0 - self._CMP_LABEL_FRAC) / len(self._CMP_COL_ORDER)

        # the two constraints, each solved for the cell font size
        total_lines = self._CMP_TEXT_LINES + self._CMP_GAP_LINES
        fs_vertical = ax_h_pt / (total_lines * self._CMP_LINE_H)
        fs_horizontal = (ax_w_pt * col_frac * 0.92) / (self._CMP_WRAP * self._CMP_EM_PER_CHAR)
        fs = max(self._CMP_FS_MIN, min(self._CMP_FS_MAX, min(fs_vertical, fs_horizontal)))

        # the footnotes are single ~92-character lines spanning the whole
        # width, so they are width-bound by a different budget than the
        # cells -- sized separately, or they run off both edges.
        fs_note = min(fs * self._CMP_NOTE_RATIO, ax_w_pt / (92 * self._CMP_EM_PER_CHAR))

        line = fs * self._CMP_LINE_H / ax_h_pt  # one line of cell text, axes fraction
        # Height the type could not use (because the *width* limited it)
        # goes into the gaps: text height is fixed by the font, so solve
        # text + spread * gaps = available.
        available_lines = 1.0 / line if line > 0 else total_lines
        spread = (available_lines - self._CMP_TEXT_LINES) / self._CMP_GAP_LINES
        spread = max(1.0, min(self._CMP_SPREAD_MAX, spread))

        return {
            "fs": fs,
            "fs_header": fs * self._CMP_HEADER_RATIO,
            "fs_note": fs_note,
            "line": line,
            "spread": spread,
            "col_frac": col_frac,
        }

    def _render_comparison_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        g = self._cmp_geometry()
        fs, fs_header, fs_note = g["fs"], g["fs_header"], g["fs_note"]
        line, sp, col_frac = g["line"], g["spread"], g["col_frac"]
        note_line = line * self._CMP_NOTE_RATIO

        cols_x = {
            name: self._CMP_LABEL_FRAC + col_frac * (i + 0.5)
            for i, name in enumerate(self._CMP_COL_ORDER)
        }
        label_x = 0.004

        table = values["table"]
        row_order = list(values["table_rows"])
        reveal_map = {
            "Representação": values["reveal_repr"],
            "Ativação": values["reveal_activation"],
            "Domínio temporal": values["reveal_domain"],
            "Treinamento": values["reveal_training"],
            "Operação principal": values["reveal_operation"],
        }

        # top-down placement cursor, in axes fractions: `text` consumes
        # what the font needs, `gap` consumes stretchable space.
        cursor = 1.0

        def gap(lines: float) -> None:
            nonlocal cursor
            cursor -= lines * line * sp

        def text(height_lines: float) -> float:
            """Consume a text block and return the y its centre sits at."""
            nonlocal cursor
            centre = cursor - height_lines * line / 2.0
            cursor -= height_lines * line
            return centre

        gap(0.1)
        header_y = text(self._CMP_HEADER_RATIO)
        for name, x in cols_x.items():
            self._fading_text(x, header_y, name, self._CMP_COL_COLORS[name], 1.0, fontsize=fs_header, weight="bold")
        # a hairline under the column heads: at this type size the table
        # needs one cue that the top row is a *header*, not just another
        # (differently coloured) row of values.
        gap(0.35)
        self._cmp_rule(cursor, 1.0)
        gap(0.5)

        for index, row_name in enumerate(row_order):
            if index:
                gap(0.45)
            reveal = float(reveal_map[row_name])
            y = text(2.0)
            # the row label stays faintly visible before its row is
            # revealed (alpha floor 0.12) so the table's full height is
            # apparent from the first checkpoint -- the same "skeleton
            # first" rule the node diagrams follow.
            self._fading_text(
                label_x, y, textwrap.fill(row_name, self._CMP_LABEL_WRAP), "black", max(reveal, 0.12),
                fontsize=fs, weight="bold", ha="left",
            )
            for name, cell in zip(self._CMP_COL_ORDER, table[row_name]):
                self._fading_text(
                    cols_x[name], y, textwrap.fill(cell, self._CMP_WRAP),
                    self._CMP_COL_COLORS[name], reveal, fontsize=fs,
                )

        reveal_outputs = float(values["reveal_outputs"])
        gap(0.55)
        self._cmp_rule(cursor, reveal_outputs)
        gap(0.45)
        outputs_y = text(1.0)
        if reveal_outputs > 0.02:
            # "Saída" spelled out: this label used to read just "O", which
            # is not an abbreviation of anything a student can decode.
            self._fading_text(
                label_x, outputs_y, "Saída", "black", reveal_outputs,
                fontsize=fs, weight="bold", ha="left",
            )
            outputs = {
                "ANN": f"y = {values['y_ann']:g}",
                "BitNet": f"y = {values['y_bitnet']:g}",
                "SNN": f"{values['snn_spike_count']} spikes",
            }
            for name, text_value in outputs.items():
                self._fading_text(cols_x[name], outputs_y, text_value, self._CMP_COL_COLORS[name], reveal_outputs, fontsize=fs)

        gap(0.5)
        note1_y = cursor - note_line / 2.0
        cursor -= note_line
        self._fading_text(
            0.5, note1_y,
            "Tipo de gradiente: exato (ANN) / STE (BitNet) / substituto (SNN) — análogos, não idênticos.",
            ACCENT_COLOR, float(values["reveal_gradients"]), fontsize=fs_note,
        )
        gap(0.3)
        note2_y = cursor - note_line / 2.0
        self._fading_text(
            0.5, note2_y,
            "Eficiência não é garantida pela arquitetura: depende de hardware, memória e workload.",
            SNN_COLOR, float(values["reveal_caveat"]), fontsize=fs_note, weight="bold",
        )

    def _cmp_rule(self, y: float, alpha: float) -> None:
        """The table's horizontal hairlines (header underline, outputs divider)."""
        if alpha <= 0.02:
            return
        self._ax.plot(
            [0.0, 1.0], [y, y], color=NEUTRAL_COLOR,
            linewidth=1.0, alpha=_SKELETON_ALPHA * alpha, solid_capstyle="butt",
        )
