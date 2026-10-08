"""Renderer for `backprop.rube_goldberg` — the chain rule as a machine.

Physical layout (pixel/data-unit positions) lives entirely here, never in
the demo module: `backprop/demos/rube_goldberg_chain.py` only ever outputs
*logical* quantities (which station is on, the ball's progress 0..5, its
signed value) — the same split the rest of this package already uses
(e.g. `mlp.py` owns every neuron's (x, y), `multilayer_network.py` only
emits which neuron is "active"). `_RG_STATION_X` below is this renderer's
one source of truth for where each of the five gadgets sits.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.widgets.renderers._painting import _FILL_ALPHA, _SKELETON_ALPHA


class RubeGoldbergRendererMixin:
    """Draws the `rube_goldberg` frame kind."""

    _RG_XLIM = (0.0, 18.5)
    _RG_YLIM = (0.0, 10.0)
    _RG_TRACK_Y = 3.0
    #: Five gadget stations, evenly spaced, in the order the demo's
    #: `CHAIN_NAMES`/`chain_values` are (funnel, ramp, lever, ramp, pulley).
    _RG_STATION_X = (2.6, 5.8, 9.0, 12.2, 15.4)
    _RG_BUCKET_X = 17.3
    #: `ball_progress` 0..4 sits at a station; 5 is the bucket. One extra
    #: slot at the end of the position table makes "ball_progress -> x"
    #: a single lookup/interpolation, bucket included, instead of a
    #: special case.
    _RG_SLOT_X = (*_RG_STATION_X, _RG_BUCKET_X)
    _RG_GADGET_Y = 6.3
    _RG_CAPTION_Y = 8.4
    _RG_CHIP_Y = 4.55
    _RG_CHIP_W, _RG_CHIP_H = 2.1, 1.0
    #: How high the ball arcs above the track mid-transition (a pure
    #: function of the already-tweened `ball_progress`, not its own stored
    #: field — see module docstring: the demo emits no geometry). Capped
    #: well under `_RG_CHIP_Y`'s bottom edge (chip center 4.55, half-height
    #: + boxstyle pad ~0.75 -> bottom ~3.8): a taller bob sends the ball
    #: straight through the already-fired factor chip it is passing under.
    _RG_BOB_HEIGHT = 0.55
    _RG_GADGET_COLORS = (NEUTRAL_COLOR, BITNET_COLOR, ACCENT_COLOR, CONVERGE_COLOR, BITNET_COLOR)
    #: What each gadget DOES, physically -- own copy, independent of the
    #: demo's GADGET_CAPTIONS (same split as chain_layers.py's _CL_CAPTIONS:
    #: the renderer's display text is this renderer's concern, not shared
    #: with the module that owns the numbers).
    _RG_GADGET_CAPTIONS = (
        "funil\nrecebe o erro",
        "rampa\n× σ'(z2)",
        "alavanca\n× w2",
        "rampa\n× σ'(z1)",
        "roldana\n× x",
    )

    # -- small shape helpers (unique to this machine, so they live here
    # rather than in the shared _painting.py vocabulary) -----------------
    def _rg_funnel(self, x: float, y: float, color: str, alpha: float) -> None:
        pts = [(x - 0.9, y + 0.7), (x + 0.9, y + 0.7), (x + 0.22, y - 0.7), (x - 0.22, y - 0.7)]
        self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5))

    def _rg_ramp(self, x: float, y: float, color: str, alpha: float, mirror: bool) -> None:
        dx = -1.0 if mirror else 1.0
        pts = [(x - dx * 0.95, y - 0.65), (x + dx * 0.95, y + 0.65), (x + dx * 0.95, y - 0.65)]
        self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5))

    def _rg_lever(self, x: float, y: float, color: str, alpha: float, flipped: bool) -> None:
        # a pivot triangle plus a bar tilted one way or the other depending
        # on the weight's sign — the one gadget whose GEOMETRY, not just its
        # label, changes with the number, so the sign flip is seen even
        # before the ball arrives.
        self._ax.add_patch(Polygon(
            [(x - 0.35, y - 0.6), (x + 0.35, y - 0.6), (x, y)],
            closed=True, facecolor=NEUTRAL_COLOR, edgecolor=NEUTRAL_COLOR, alpha=alpha * 0.8, linewidth=1.2,
        ))
        tilt = 0.45 if flipped else -0.45
        self._ax.plot(
            [x - 1.1, x + 1.1], [y + tilt, y - tilt],
            color=color, linewidth=5.0, alpha=alpha, solid_capstyle="round",
        )

    def _rg_pulley(self, x: float, y: float, color: str, alpha: float) -> None:
        self._ax.add_patch(Circle((x, y), 0.55, facecolor="none", edgecolor=color, linewidth=3.0, alpha=alpha))
        self._ax.add_patch(Circle((x, y), 0.1, facecolor=color, edgecolor=color, alpha=alpha))
        for k in range(4):
            angle = k * math.pi / 2
            self._ax.plot(
                [x, x + 0.55 * math.cos(angle)], [y, y + 0.55 * math.sin(angle)],
                color=color, linewidth=1.6, alpha=alpha,
            )

    def _rg_bucket(self, x: float, y: float, color: str, alpha: float) -> None:
        pts = [(x - 0.8, y + 0.6), (x + 0.8, y + 0.6), (x + 0.55, y - 0.5), (x - 0.55, y - 0.5)]
        self._ax.add_patch(Polygon(
            pts, closed=True, facecolor="none", edgecolor=color, alpha=max(alpha, _SKELETON_ALPHA), linewidth=2.2,
        ))
        if alpha > 0.02:
            self._ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", alpha=_FILL_ALPHA * alpha))

    _RG_GADGET_DRAWERS = ("_rg_funnel", "_rg_ramp", "_rg_lever", "_rg_ramp", "_rg_pulley")

    def _rg_draw_gadget(self, index: int, x: float, color: str, alpha: float, w2_value: float) -> None:
        name = self._RG_GADGET_DRAWERS[index]
        if name == "_rg_ramp":
            self._rg_ramp(x, self._RG_GADGET_Y, color, alpha, mirror=(index == 3))
        elif name == "_rg_lever":
            self._rg_lever(x, self._RG_GADGET_Y, color, alpha, flipped=(w2_value < 0))
        elif name == "_rg_funnel":
            self._rg_funnel(x, self._RG_GADGET_Y, color, alpha)
        else:
            self._rg_pulley(x, self._RG_GADGET_Y, color, alpha)

    def _rg_ball_position(self, progress: float) -> tuple[float, float]:
        progress = max(0.0, min(float(progress), len(self._RG_SLOT_X) - 1))
        i = min(int(progress), len(self._RG_SLOT_X) - 2)
        frac = progress - i
        x0, x1 = self._RG_SLOT_X[i], self._RG_SLOT_X[i + 1]
        x = x0 + (x1 - x0) * frac
        bob = math.sin(frac * math.pi) * self._RG_BOB_HEIGHT
        return x, self._RG_TRACK_Y + bob

    def _render_rube_goldberg(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=self._RG_XLIM, ylim=self._RG_YLIM)

        chain_names = values["chain_names"]
        chain_values = values["chain_values"]
        station_fired = np.asarray(values["station_fired"], dtype=float)
        station_glow = np.asarray(values["station_glow"], dtype=float)
        w2_value = float(chain_values[2])
        n_stations = len(chain_names)

        # the track itself: a static line joining every station to the
        # bucket, so the whole machine's shape is visible before the first
        # click -- "skeleton first", same rule every other renderer follows.
        self._ax.plot(
            [self._RG_SLOT_X[0] - 1.0, self._RG_BUCKET_X], [self._RG_TRACK_Y] * 2,
            color=NEUTRAL_COLOR, linewidth=2.0, alpha=_SKELETON_ALPHA, zorder=0,
        )
        for i in range(n_stations):
            fill = float(station_fired[i])
            a = (self._RG_SLOT_X[i], self._RG_TRACK_Y)
            b = (self._RG_SLOT_X[i + 1], self._RG_TRACK_Y)
            self._flow_arrow(a, b, fill, NEUTRAL_COLOR)

        for i in range(n_stations):
            x = self._RG_STATION_X[i]
            glow = float(station_glow[i])
            base_color = self._RG_GADGET_COLORS[i]
            alpha = max(_SKELETON_ALPHA, 0.9 if glow > 0.5 else 0.65)
            self._rg_draw_gadget(i, x, base_color, alpha, w2_value)
            self._fading_text(x, self._RG_CAPTION_Y, self._RG_GADGET_CAPTIONS[i], NEUTRAL_COLOR, 0.85, fontsize=10)
            fired = float(station_fired[i])
            if glow > 0.5:
                self._ax.add_patch(Circle((x, self._RG_GADGET_Y), 1.25, facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.22, zorder=0))
            if fired > 0.02:
                self._box(
                    x, self._RG_CHIP_Y, f"{chain_names[i]}\n{float(chain_values[i]):+.4f}",
                    ACCENT_COLOR, alpha=fired, w=self._RG_CHIP_W, h=self._RG_CHIP_H, fontsize=10,
                )

        bucket_reveal = float(values.get("bucket_reveal", 0.0))
        self._rg_bucket(self._RG_BUCKET_X, self._RG_GADGET_Y - 0.3, SNN_COLOR, bucket_reveal)
        self._fading_text(self._RG_BUCKET_X, self._RG_CAPTION_Y, "balde\n∂L/∂w1", NEUTRAL_COLOR, 0.85, fontsize=10)
        if bucket_reveal > 0.02:
            g_w1 = float(values["g_w1"])
            self._fading_text(
                self._RG_BUCKET_X, self._RG_CHIP_Y, f"∂L/∂w1 = {g_w1:+.6f}",
                SNN_COLOR, bucket_reveal, fontsize=11, weight="bold",
            )

        ball_reveal = float(values.get("ball_reveal", 0.0))
        if ball_reveal > 0.02:
            progress = float(values.get("ball_progress", 0.0))
            ball_value = float(values.get("ball_value", 0.0))
            x, y = self._rg_ball_position(progress)
            # schematic size only (sqrt of a clamped magnitude) -- the
            # EXACT value is the label, never implied by the radius alone
            # (NO_FALSE_FLUENCY: a picture must not claim more precision
            # than the number printed next to it).
            radius = max(0.22, min(0.75, 0.25 + 0.55 * math.sqrt(min(abs(ball_value), 1.2))))
            color = BITNET_COLOR if ball_value >= 0 else SNN_COLOR
            self._ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=color, alpha=ball_reveal * _FILL_ALPHA + 0.2, zorder=5))
            # below the ball, not above: above is where the factor chip
            # sits (_RG_CHIP_Y), and the two texts collided there.
            self._ax.text(
                x, y - radius - 0.3, f"{ball_value:+.4f}", ha="center", va="top",
                fontsize=10, color=color, alpha=ball_reveal, fontweight="bold", zorder=5,
            )

        self._fading_text(
            (self._RG_XLIM[0] + self._RG_XLIM[1]) / 2, 0.55,
            "clique na tela (ou “Próximo”) para empurrar a bolinha\n"
            "tamanho da bolinha é ilustrativo, não em escala",
            NEUTRAL_COLOR, 0.65, fontsize=8.5,
        )
