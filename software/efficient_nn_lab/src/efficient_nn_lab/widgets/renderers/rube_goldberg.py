"""Renderer for `backprop.rube_goldberg` — the chain rule as a machine.

Every shape this renderer draws for the track, ramps, lever, pulley, and
bucket comes from `backprop/demos/rube_goldberg_physics.py`'s own geometry
constants and the frame's `lever_angle`/`pulley_angle`/`ball_x`/`ball_y`
fields — never a hand-picked position. That is deliberate: the picture must
never be able to disagree with what the ball actually collided with in the
simulation. The only things this module still owns independently are pure
display concerns the physics has no opinion about: colors, caption text,
and the chip boxes that show each factor's name/value once revealed.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, Polygon

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.backprop.demos.rube_goldberg_physics import (
    BALL_RADIUS,
    BUCKET_RIGHT_WALL_X,
    BUCKET_WALL_TOP,
    LEVER_HALF_LEN,
    LEVER_PIVOT,
    P_BUCKET_DROP_TOP,
    P_BUCKET_FLOOR_LEFT,
    P_FLOOR_START,
    P_FUNNEL_TOP,
    P_LEVER_LEFT,
    P_LEVER_RIGHT,
    P_RAMP1_BOTTOM,
    P_RAMP1_TOP,
    P_RAMP2_BOTTOM,
    P_RAMP2_TOP,
    P_SPAWN,
    PULLEY_CENTER,
    PULLEY_RADIUS,
)
from efficient_nn_lab.widgets.renderers._painting import _FILL_ALPHA, _SKELETON_ALPHA


class RubeGoldbergRendererMixin:
    """Draws the `rube_goldberg` frame kind."""

    _RG_XLIM = (1.3, 19.3)
    _RG_YLIM = (-0.6, 8.3)
    _RG_CAPTION_Y = 7.9
    _RG_CHIP_W, _RG_CHIP_H = 2.0, 0.95
    #: One (x, y) per gadget, in `CHAIN_NAMES` order (funnel+ramp1, lever,
    #: ramp2, pulley, bucket) — where that gadget's chip/glow is drawn. Each
    #: sits a fixed clearance above the real track height at that x, read
    #: directly off the physics module's own waypoints.
    _RG_CHIP_XY = (
        (P_SPAWN[0], 6.9),
        ((P_RAMP1_TOP[0] + P_RAMP1_BOTTOM[0]) / 2, 6.3),
        (LEVER_PIVOT[0], 5.3),
        ((P_RAMP2_TOP[0] + P_RAMP2_BOTTOM[0]) / 2, 4.4),
        (PULLEY_CENTER[0], 3.5),
    )
    _RG_GADGET_COLORS = (NEUTRAL_COLOR, BITNET_COLOR, ACCENT_COLOR, CONVERGE_COLOR, BITNET_COLOR)
    _RG_GADGET_CAPTIONS = (
        "funil\nrecebe o erro",
        "rampa\n× σ'(z2)",
        "alavanca\n× w2",
        "rampa\n× σ'(z1)",
        "roldana\n× x",
    )

    # -- static track (skeleton-first: drawn every frame, faint, so the
    # whole machine's shape is visible before the first click) -------------
    def _rg_track_skeleton(self) -> None:
        segments = (
            (P_FUNNEL_TOP, P_RAMP1_TOP),
            (P_RAMP1_TOP, P_RAMP1_BOTTOM),
            (P_RAMP1_BOTTOM, P_LEVER_LEFT),
            (P_LEVER_RIGHT, P_RAMP2_TOP),
            (P_RAMP2_TOP, P_RAMP2_BOTTOM),
            (P_FLOOR_START, P_BUCKET_DROP_TOP),
            (P_BUCKET_DROP_TOP, P_BUCKET_FLOOR_LEFT),
            (P_BUCKET_FLOOR_LEFT, (BUCKET_RIGHT_WALL_X, P_BUCKET_FLOOR_LEFT[1])),
            ((BUCKET_RIGHT_WALL_X, P_BUCKET_FLOOR_LEFT[1]), (BUCKET_RIGHT_WALL_X, BUCKET_WALL_TOP)),
        )
        for a, b in segments:
            self._ax.plot([a[0], b[0]], [a[1], b[1]], color=NEUTRAL_COLOR, linewidth=2.2, alpha=_SKELETON_ALPHA, zorder=0, solid_capstyle="round")

    def _rg_funnel(self, color: str, alpha: float) -> None:
        x, y = P_FUNNEL_TOP[0], P_FUNNEL_TOP[1] + 0.35
        pts = [(x - 0.85, y + 0.55), (x + 0.85, y + 0.55), (x + 0.2, y - 0.55), (x - 0.2, y - 0.55)]
        self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5, zorder=1))

    def _rg_lever(self, lever_angle: float, color: str, alpha: float) -> None:
        px, py = LEVER_PIVOT
        dx = LEVER_HALF_LEN * math.cos(lever_angle)
        dy = LEVER_HALF_LEN * math.sin(lever_angle)
        self._ax.plot([px - dx, px + dx], [py - dy, py + dy], color=color, linewidth=6.0, alpha=alpha, solid_capstyle="round", zorder=2)
        self._ax.add_patch(Polygon(
            [(px - 0.3, py - 0.5), (px + 0.3, py - 0.5), (px, py)],
            closed=True, facecolor=NEUTRAL_COLOR, edgecolor=NEUTRAL_COLOR, alpha=alpha * 0.8, linewidth=1.0, zorder=1,
        ))

    def _rg_pulley(self, pulley_angle: float, color: str, alpha: float) -> None:
        x, y = PULLEY_CENTER
        self._ax.add_patch(Circle((x, y), PULLEY_RADIUS, facecolor="none", edgecolor=color, linewidth=3.0, alpha=alpha, zorder=2))
        self._ax.add_patch(Circle((x, y), 0.07, facecolor=color, edgecolor=color, alpha=alpha, zorder=2))
        for k in range(4):
            angle = pulley_angle + k * math.pi / 2
            self._ax.plot(
                [x, x + PULLEY_RADIUS * math.cos(angle)], [y, y + PULLEY_RADIUS * math.sin(angle)],
                color=color, linewidth=1.6, alpha=alpha, zorder=2,
            )

    def _rg_bucket(self, color: str, alpha: float) -> None:
        floor_y = P_BUCKET_FLOOR_LEFT[1]
        left_x = P_BUCKET_FLOOR_LEFT[0]
        pts = [(left_x, floor_y), (BUCKET_RIGHT_WALL_X, floor_y), (BUCKET_RIGHT_WALL_X, BUCKET_WALL_TOP), (left_x, floor_y + 1.1)]
        self._ax.add_patch(Polygon(
            pts, closed=False, facecolor="none", edgecolor=color, alpha=max(alpha, _SKELETON_ALPHA), linewidth=2.2, zorder=1,
        ))
        if alpha > 0.02:
            self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", alpha=_FILL_ALPHA * alpha, zorder=0))

    _RG_GADGET_DRAWERS = ("funnel", "ramp1", "lever", "ramp2", "pulley")

    def _rg_draw_gadget(self, index: int, color: str, alpha: float, lever_angle: float, pulley_angle: float) -> None:
        kind = self._RG_GADGET_DRAWERS[index]
        if kind == "ramp1":
            self._ax.plot(
                [P_RAMP1_TOP[0], P_RAMP1_BOTTOM[0]], [P_RAMP1_TOP[1], P_RAMP1_BOTTOM[1]],
                color=color, linewidth=5.0, alpha=alpha, solid_capstyle="round", zorder=1,
            )
        elif kind == "ramp2":
            self._ax.plot(
                [P_RAMP2_TOP[0], P_RAMP2_BOTTOM[0]], [P_RAMP2_TOP[1], P_RAMP2_BOTTOM[1]],
                color=color, linewidth=5.0, alpha=alpha, solid_capstyle="round", zorder=1,
            )
        elif kind == "lever":
            self._rg_lever(lever_angle, color, alpha)
        elif kind == "funnel":
            self._rg_funnel(color, alpha)
        else:
            self._rg_pulley(pulley_angle, color, alpha)

    def _render_rube_goldberg(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=self._RG_XLIM, ylim=self._RG_YLIM)

        chain_names = values["chain_names"]
        chain_values = values["chain_values"]
        station_fired = np.asarray(values["station_fired"], dtype=float)
        station_glow = np.asarray(values["station_glow"], dtype=float)
        lever_angle = float(values.get("lever_angle", 0.0))
        pulley_angle = float(values.get("pulley_angle", 0.0))
        n_stations = len(chain_names)

        self._rg_track_skeleton()

        for i in range(n_stations):
            cx, cy = self._RG_CHIP_XY[i]
            glow = float(station_glow[i])
            base_color = self._RG_GADGET_COLORS[i]
            alpha = max(_SKELETON_ALPHA, 0.9 if glow > 0.5 else 0.65)
            self._rg_draw_gadget(i, base_color, alpha, lever_angle, pulley_angle)
            self._fading_text(cx, self._RG_CAPTION_Y, self._RG_GADGET_CAPTIONS[i], NEUTRAL_COLOR, 0.85, fontsize=10)
            fired = float(station_fired[i])
            if glow > 0.5:
                self._ax.add_patch(Circle((cx, cy - 0.3), 1.15, facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.2, zorder=0))
            if fired > 0.02:
                self._box(
                    cx, cy, f"{chain_names[i]}\n{float(chain_values[i]):+.4f}",
                    ACCENT_COLOR, alpha=fired, w=self._RG_CHIP_W, h=self._RG_CHIP_H, fontsize=10,
                )

        bucket_reveal = float(values.get("bucket_reveal", 0.0))
        self._rg_bucket(SNN_COLOR, bucket_reveal)
        bucket_cx = (P_BUCKET_FLOOR_LEFT[0] + BUCKET_RIGHT_WALL_X) / 2
        self._fading_text(bucket_cx, self._RG_CAPTION_Y, "balde\n∂L/∂w1", NEUTRAL_COLOR, 0.85, fontsize=10)
        if bucket_reveal > 0.02:
            g_w1 = float(values["g_w1"])
            self._fading_text(
                bucket_cx, BUCKET_WALL_TOP + 0.5, f"∂L/∂w1 = {g_w1:+.6f}",
                SNN_COLOR, bucket_reveal, fontsize=11, weight="bold",
            )

        ball_reveal = float(values.get("ball_reveal", 0.0))
        if ball_reveal > 0.02:
            x = float(values.get("ball_x", P_SPAWN[0]))
            y = float(values.get("ball_y", P_SPAWN[1]))
            ball_angle = float(values.get("ball_angle", 0.0))
            ball_value = float(values.get("ball_value", 0.0))
            # display radius is a fixed, honest match to the simulated
            # ball's own physics radius (NO_FALSE_FLUENCY: the picture must
            # not claim a size-encodes-magnitude relationship it doesn't
            # have now that a real physics radius exists to be honest about).
            radius = BALL_RADIUS * 1.15
            color = BITNET_COLOR if ball_value >= 0 else SNN_COLOR
            self._ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=color, alpha=ball_reveal * _FILL_ALPHA + 0.25, zorder=5))
            # a short radial mark showing the ball's own simulated rotation
            # -- real spin from rolling contact, not a decorative spinner.
            self._ax.plot(
                [x, x + radius * math.cos(ball_angle)], [y, y + radius * math.sin(ball_angle)],
                color="white", linewidth=1.6, alpha=ball_reveal, zorder=6,
            )
            self._ax.text(
                x, y - radius - 0.28, f"{ball_value:+.4f}", ha="center", va="top",
                fontsize=10, color=color, alpha=ball_reveal, fontweight="bold", zorder=5,
            )

        self._fading_text(
            (self._RG_XLIM[0] + self._RG_XLIM[1]) / 2, -0.3,
            "clique na tela (ou “Próximo”) para empurrar a bolinha — "
            "física real: a bolinha rola, a alavanca tomba, a roldana gira",
            NEUTRAL_COLOR, 0.65, fontsize=8.5,
        )
