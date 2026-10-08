"""Renderer for `backprop.rube_goldberg` — the chain rule as a machine.

Every shape this renderer draws for the track, ramps, lever, pulley, and
bucket comes from the frame's own ``values["layout"]`` (a
`rube_goldberg_physics.MachineLayout`, built fresh by `simulate_machine`
for the demo's CURRENT w2/x) and the frame's
`lever_angle`/`pulley_angle`/`ball_x`/`ball_y` fields — never a hand-picked
position. That is deliberate: the picture must never be able to disagree
with what the ball actually collided with in the simulation, and since
`layout` now itself changes whenever w2/x change, drawing from fixed
module-level constants (the earlier version's approach) would silently go
stale the first time someone dragged a handle. The only things this module
still owns independently are pure display concerns the physics has no
opinion about: colors, caption text, drag-handle hit-testing, and the chip
boxes that show each factor's name/value once revealed.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, Polygon

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.backprop.demos.rube_goldberg_physics import (
    BALL_RADIUS,
    LEVER_HALF_LEN,
    PULLEY_RADIUS,
)
from efficient_nn_lab.widgets.renderers._painting import _FILL_ALPHA, _SKELETON_ALPHA


class RubeGoldbergRendererMixin:
    """Draws the `rube_goldberg` frame kind."""

    _RG_CAPTION_DY = 1.1  # caption row sits this far above the funnel's drop point
    _RG_CHIP_W, _RG_CHIP_H = 2.0, 0.95
    _RG_GADGET_COLORS = (NEUTRAL_COLOR, BITNET_COLOR, ACCENT_COLOR, CONVERGE_COLOR, BITNET_COLOR)
    _RG_GADGET_CAPTIONS = (
        "funil\nrecebe o erro",
        "rampa\n× σ'(z2)",
        "alavanca\n× w2\n(arraste)",
        "rampa\n× σ'(z1)",
        "roldana\n× x\n(arraste)",
    )
    #: Drag-handle hit radius, in data units -- generous (bigger than the
    #: lever beam or pulley wheel themselves) so grabbing either feels easy
    #: rather than pixel-precise. See `rg_handle_at`.
    _RG_HANDLE_HIT_RADIUS = 1.0
    #: Data-units of vertical drag that sweeps a handle's full value range
    #: -- the drag is relative (delta from press, not an absolute position
    #: mapping), so the value never jumps the moment you grab a handle.
    _RG_DRAG_SPAN = 3.0

    #: Every piece of the machine -- connectors AND the ramps/floor the
    #: gadget overlays also draw over -- shares this one stroke weight, so
    #: the rail reads as one continuous mechanism. An earlier version drew
    #: the connectors as a separate, much thinner/fainter "skeleton" line
    #: than the bold gadget overlays (5-6pt): at every hand-off the bold
    #: piece visibly thinned to a pale thread before the next bold piece
    #: picked up, which read as disconnected floating objects even though
    #: the endpoints were coordinate-exact (confirmed by rendering and
    #: cropping each junction -- a real bug report, not a false alarm).
    _RG_RAIL_LINEWIDTH = 5.0
    _RG_RAIL_ALPHA = 0.55

    def rg_handle_at(self, values: dict[str, object], data_x: float, data_y: float) -> str | None:
        """Which draggable parameter (``"w2"``, ``"x"``, or ``None``) sits
        under a point in data coordinates -- shared by `widgets/neuron_view.py`'s
        mouse-press handler so hit-testing lives next to the geometry it
        tests against instead of being duplicated/guessed elsewhere."""
        layout = values.get("layout")
        if layout is None:
            return None
        r2 = self._RG_HANDLE_HIT_RADIUS ** 2
        lx, ly = layout.lever_pivot
        if (data_x - lx) ** 2 + (data_y - ly) ** 2 <= r2:
            return "w2"
        px, py = layout.pulley_center
        if (data_x - px) ** 2 + (data_y - py) ** 2 <= r2:
            return "x"
        return None

    def _rg_bounds(self, layout) -> tuple[tuple[float, float], tuple[float, float]]:
        """xlim/ylim that fit this layout with margin -- computed fresh per
        layout (rather than a fixed constant) because steeper ramps/a
        bigger lever tilt genuinely change how tall/wide the machine is."""
        xs = [
            layout.funnel_top[0] - 1.0, layout.funnel_top[0] + 1.0,
            layout.ramp1_top[0], layout.ramp1_bottom[0],
            layout.lever_left[0], layout.lever_right[0],
            layout.ramp2_top[0], layout.ramp2_bottom[0],
            layout.bucket_drop_top[0], layout.bucket_right_wall_x + 0.6,
        ]
        ys = [
            layout.funnel_top[1] + 2.0,  # headroom for the ball's spawn/drop
            layout.ramp1_top[1], layout.ramp1_bottom[1],
            layout.lever_left[1], layout.lever_right[1],
            layout.ramp2_top[1], layout.ramp2_bottom[1],
            layout.bucket_floor_left[1] - 0.6, layout.bucket_wall_top + 0.9,
        ]
        return (min(xs) - 0.3, max(xs) + 0.3), (min(ys) - 0.3, max(ys) + 0.3)

    # -- static track (rail-first: drawn every frame at full stroke weight,
    # so the whole machine's shape reads as one connected mechanism before
    # the first click, not a diagram of separate labelled parts) -----------
    def _rg_track_skeleton(self, layout) -> None:
        segments = (
            (layout.funnel_top, layout.ramp1_top),
            (layout.ramp1_top, layout.ramp1_bottom),
            (layout.ramp1_bottom, layout.lever_left),
            (layout.lever_right, layout.ramp2_top),
            (layout.ramp2_top, layout.ramp2_bottom),
            (layout.ramp2_bottom, layout.bucket_drop_top),
            (layout.bucket_drop_top, layout.bucket_floor_left),
            (layout.bucket_floor_left, (layout.bucket_right_wall_x, layout.bucket_floor_left[1])),
            ((layout.bucket_right_wall_x, layout.bucket_floor_left[1]), (layout.bucket_right_wall_x, layout.bucket_wall_top)),
        )
        for a, b in segments:
            self._ax.plot(
                [a[0], b[0]], [a[1], b[1]], color=NEUTRAL_COLOR, linewidth=self._RG_RAIL_LINEWIDTH,
                alpha=self._RG_RAIL_ALPHA, zorder=0, solid_capstyle="round",
            )

    def _rg_funnel(self, layout, color: str, alpha: float) -> None:
        # bottom opening centred exactly ON funnel_top (the rail's own
        # start point) rather than above it, so the spout and the rail
        # meet at one coincident point instead of the rail emerging from
        # partway up the funnel's sloped side.
        x, y_bottom = layout.funnel_top
        y_top = y_bottom + 1.1
        pts = [(x - 0.85, y_top), (x + 0.85, y_top), (x + 0.15, y_bottom), (x - 0.15, y_bottom)]
        self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5, zorder=1))

    def _rg_handle_grip(self, x: float, y: float, color: str, alpha: float) -> None:
        """A small dashed ring marking a draggable handle -- distinct from
        every other (fixed) gadget so the two interactive pieces read as
        "grab me" at a glance."""
        self._ax.add_patch(Circle(
            (x, y), 0.3, facecolor="none", edgecolor=color, alpha=alpha, linewidth=1.6,
            linestyle=(0, (2, 2)), zorder=3,
        ))

    def _rg_lever(self, layout, lever_angle: float, color: str, alpha: float) -> None:
        px, py = layout.lever_pivot
        dx = LEVER_HALF_LEN * math.cos(lever_angle)
        dy = LEVER_HALF_LEN * math.sin(lever_angle)
        self._ax.plot([px - dx, px + dx], [py - dy, py + dy], color=color, linewidth=6.0, alpha=alpha, solid_capstyle="round", zorder=2)
        self._ax.add_patch(Polygon(
            [(px - 0.3, py - 0.5), (px + 0.3, py - 0.5), (px, py)],
            closed=True, facecolor=NEUTRAL_COLOR, edgecolor=NEUTRAL_COLOR, alpha=alpha * 0.8, linewidth=1.0, zorder=1,
        ))
        self._rg_handle_grip(px, py, color, alpha)

    def _rg_pulley(self, layout, pulley_angle: float, color: str, alpha: float) -> None:
        x, y = layout.pulley_center
        self._ax.add_patch(Circle((x, y), PULLEY_RADIUS, facecolor="none", edgecolor=color, linewidth=3.0, alpha=alpha, zorder=2))
        self._ax.add_patch(Circle((x, y), 0.07, facecolor=color, edgecolor=color, alpha=alpha, zorder=2))
        for k in range(4):
            angle = pulley_angle + k * math.pi / 2
            self._ax.plot(
                [x, x + PULLEY_RADIUS * math.cos(angle)], [y, y + PULLEY_RADIUS * math.sin(angle)],
                color=color, linewidth=1.6, alpha=alpha, zorder=2,
            )
        self._rg_handle_grip(x, y, color, alpha)

    def _rg_bucket(self, layout, color: str, alpha: float) -> None:
        floor_y = layout.bucket_floor_left[1]
        left_x = layout.bucket_floor_left[0]
        pts = [(left_x, floor_y), (layout.bucket_right_wall_x, floor_y), (layout.bucket_right_wall_x, layout.bucket_wall_top), (left_x, floor_y + 1.1)]
        self._ax.add_patch(Polygon(
            pts, closed=False, facecolor="none", edgecolor=color, alpha=max(alpha, self._RG_RAIL_ALPHA),
            linewidth=self._RG_RAIL_LINEWIDTH, zorder=1,
        ))
        if alpha > 0.02:
            self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", alpha=_FILL_ALPHA * alpha, zorder=0))

    _RG_GADGET_DRAWERS = ("funnel", "ramp1", "lever", "ramp2", "pulley")

    def _rg_draw_gadget(self, layout, index: int, color: str, alpha: float, lever_angle: float, pulley_angle: float) -> None:
        kind = self._RG_GADGET_DRAWERS[index]
        if kind == "ramp1":
            self._ax.plot(
                [layout.ramp1_top[0], layout.ramp1_bottom[0]], [layout.ramp1_top[1], layout.ramp1_bottom[1]],
                color=color, linewidth=5.0, alpha=alpha, solid_capstyle="round", zorder=1,
            )
        elif kind == "ramp2":
            self._ax.plot(
                [layout.ramp2_top[0], layout.ramp2_bottom[0]], [layout.ramp2_top[1], layout.ramp2_bottom[1]],
                color=color, linewidth=5.0, alpha=alpha, solid_capstyle="round", zorder=1,
            )
        elif kind == "lever":
            self._rg_lever(layout, lever_angle, color, alpha)
        elif kind == "funnel":
            self._rg_funnel(layout, color, alpha)
        else:
            self._rg_pulley(layout, pulley_angle, color, alpha)

    def _render_rube_goldberg(self, values: dict[str, object]) -> None:
        layout = values["layout"]
        xlim, ylim = self._rg_bounds(layout)
        self._reset_axes(xlim=xlim, ylim=ylim)
        caption_y = layout.funnel_top[1] + self._RG_CAPTION_DY

        chain_names = values["chain_names"]
        chain_values = values["chain_values"]
        station_fired = np.asarray(values["station_fired"], dtype=float)
        station_glow = np.asarray(values["station_glow"], dtype=float)
        lever_angle = float(values.get("lever_angle", 0.0))
        pulley_angle = float(values.get("pulley_angle", 0.0))
        n_stations = len(chain_names)

        chip_xy = (
            (layout.funnel_top[0], layout.funnel_top[1] + 0.3),
            ((layout.ramp1_top[0] + layout.ramp1_bottom[0]) / 2, (layout.ramp1_top[1] + layout.ramp1_bottom[1]) / 2 + 0.55),
            (layout.lever_pivot[0], layout.lever_pivot[1] + 0.95),
            ((layout.ramp2_top[0] + layout.ramp2_bottom[0]) / 2, (layout.ramp2_top[1] + layout.ramp2_bottom[1]) / 2 + 0.55),
            (layout.pulley_center[0], layout.pulley_center[1] + 0.9),
        )

        self._rg_track_skeleton(layout)

        for i in range(n_stations):
            cx, cy = chip_xy[i]
            glow = float(station_glow[i])
            base_color = self._RG_GADGET_COLORS[i]
            alpha = max(_SKELETON_ALPHA, 0.9 if glow > 0.5 else 0.65)
            self._rg_draw_gadget(layout, i, base_color, alpha, lever_angle, pulley_angle)
            self._fading_text(cx, caption_y, self._RG_GADGET_CAPTIONS[i], NEUTRAL_COLOR, 0.85, fontsize=10)
            fired = float(station_fired[i])
            if glow > 0.5:
                self._ax.add_patch(Circle((cx, cy - 0.3), 1.15, facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.2, zorder=0))
            if fired > 0.02:
                self._box(
                    cx, cy, f"{chain_names[i]}\n{float(chain_values[i]):+.4f}",
                    ACCENT_COLOR, alpha=fired, w=self._RG_CHIP_W, h=self._RG_CHIP_H, fontsize=10,
                )

        bucket_reveal = float(values.get("bucket_reveal", 0.0))
        self._rg_bucket(layout, SNN_COLOR, bucket_reveal)
        bucket_cx = (layout.bucket_floor_left[0] + layout.bucket_right_wall_x) / 2
        self._fading_text(bucket_cx, caption_y, "balde\n∂L/∂w1", NEUTRAL_COLOR, 0.85, fontsize=10)
        if bucket_reveal > 0.02:
            g_w1 = float(values["g_w1"])
            self._fading_text(
                bucket_cx, layout.bucket_wall_top + 0.5, f"∂L/∂w1 = {g_w1:+.6f}",
                SNN_COLOR, bucket_reveal, fontsize=11, weight="bold",
            )

        ball_reveal = float(values.get("ball_reveal", 0.0))
        if ball_reveal > 0.02:
            x = float(values.get("ball_x", layout.funnel_top[0]))
            y = float(values.get("ball_y", layout.funnel_top[1]))
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
            (xlim[0] + xlim[1]) / 2, ylim[0] + 0.3,
            "clique para empurrar a bolinha — arraste a alavanca (w2) ou a roldana (x) "
            "(tracejado) para mudar a rede",
            NEUTRAL_COLOR, 0.65, fontsize=8.5,
        )
