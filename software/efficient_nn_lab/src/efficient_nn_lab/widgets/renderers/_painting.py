"""The drawing vocabulary every renderer shares.

Skeleton-first is the rule these primitives exist to enforce: a demo draws
every box and arrow it will *ever* use up front, faint and unlabeled
(`_skeleton_box`, `_skeleton_arrow`), so a first-time viewer sees the shape
of the whole computation before a single number appears. The revealed
content is then painted on top with opacity and arrow-fill driven by
continuous 0..1 fields that arrive already interpolated (see core/demo.py's
tweening). Nothing is erased and redrawn as something unrelated.

Split out of the 1823-line neuron_view.py, which had grown to hold every
demo's renderer in one class. The mixins in this package carry one demo
each; this module carries what all of them speak.
"""

from __future__ import annotations

from math import atan2, degrees

import numpy as np
from matplotlib.patches import FancyBboxPatch

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.widgets._mpl_perf import fast_clear

_BOX_STYLE = dict(boxstyle="round,pad=0.25", linewidth=2.4)
_SKELETON_ALPHA = 0.35
_FILL_ALPHA = 0.55


class PaintingMixin:
    """Primitives shared by every renderer: boxes, arrows, text, insets."""

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
