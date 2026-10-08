"""Renders time-series demos: raw signal + spike raster, or input current +
membrane potential + spike raster. Both spike_generation and lif_dynamics
reveal their trace progressively as the user steps, so the x-axis is fixed
to the full-length window from the first frame instead of auto-rescaling.
A marker at the trace's current tip makes "where are we right now" always
obvious, especially mid-animation.
"""

from __future__ import annotations

from math import atan2, degrees

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR
from efficient_nn_lab.widgets._mpl_perf import fast_clear


class SignalView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 3.6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax_top, self._ax_bottom = self._figure.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        self._default_top_pos = self._ax_top.get_position()
        self._default_bottom_pos = self._ax_bottom.get_position()
        # cached, not recreated every frame -- see widgets/_mpl_perf.py.
        self._inset_ax = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, values: dict[str, object]) -> None:
        fast_clear(self._ax_top)
        fast_clear(self._ax_bottom)
        if self._inset_ax is not None:
            self._inset_ax.set_visible(False)
        kind = values.get("kind")
        if kind == "backprop_convergence":
            # make room on the right for the sigmoid inset; every other
            # kind keeps the original full-width two-panel layout.
            self._ax_top.set_position([0.11, 0.56, 0.53, 0.38])
            self._ax_bottom.set_position([0.11, 0.13, 0.53, 0.33])
        elif kind == "poisson_image_coding":
            # both panels are images of the same pixel grid -- the default
            # 2:1 height split (sized for a line chart + a thin raster
            # strip) would squash this one relative to the other, so give
            # them equal height instead.
            self._ax_top.set_position([0.06, 0.52, 0.9, 0.37])
            self._ax_bottom.set_position([0.06, 0.06, 0.9, 0.37])
        elif kind == "firing_rate_reg":
            # one panel per failure mode (dead / bursting), equal weight.
            self._ax_top.set_position([0.11, 0.56, 0.85, 0.36])
            self._ax_bottom.set_position([0.11, 0.12, 0.85, 0.36])
        else:
            self._ax_top.set_position(self._default_top_pos)
            self._ax_bottom.set_position(self._default_bottom_pos)
        if kind == "lif_trace":
            self._render_lif(values)
        elif kind == "signal_spikes":
            self._render_signal_spikes(values)
        elif kind == "poisson_spikes":
            self._render_poisson_spikes(values)
        elif kind == "poisson_image_coding":
            self._render_poisson_image(values)
        elif kind == "backprop_convergence":
            self._render_backprop_convergence(values)
        elif kind == "firing_rate_reg":
            self._render_firing_rate_reg(values)
        elif kind == "encoding_loss_mismatch":
            self._render_encoding_loss_mismatch(values)
        else:
            self._ax_top.text(0.5, 0.5, "(sem sinal para este passo)", ha="center", va="center")
        self._canvas.draw_idle()

    def _render_signal_spikes(self, values: dict[str, object]) -> None:
        signal = np.asarray(values["signal"])
        spikes = np.asarray(values["spikes"])
        n_total = int(values["n_total"])
        level = float(values["level"])
        t = np.arange(len(signal))

        self._ax_top.plot(t, signal, color=SNN_COLOR, linewidth=2)
        if len(t):
            self._ax_top.plot([t[-1]], [signal[-1]], marker="o", markersize=7, color=SNN_COLOR, zorder=4)
            self._ax_top.text(
                t[-1], signal[-1] + 0.12, f"{signal[-1]:.2f}", ha="center", fontsize=8, color=SNN_COLOR,
            )
        self._ax_top.axhline(level, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label=f"nível de disparo = {level:.2f}")
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(-1.1, 1.1)
        self._ax_top.set_ylabel("amplitude")
        self._ax_top.legend(loc="upper right", fontsize=8)

        spike_times = t[spikes > 0]
        for st in spike_times:
            self._ax_bottom.text(st, 1.05, f"t={st}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.vlines(spike_times, 0, 1, color=SNN_COLOR, linewidth=2)
        self._ax_bottom.set_yscale("linear")  # backprop_convergence leaves this axis log-scaled otherwise
        self._ax_bottom.set_xlim(0, n_total)
        self._ax_bottom.set_ylim(0, 1.2)
        self._ax_bottom.set_yticks([])
        self._ax_bottom.set_xlabel("tempo (passos)")
        self._ax_bottom.set_ylabel("spikes")

    def _render_poisson_spikes(self, values: dict[str, object]) -> None:
        signal = np.asarray(values["signal"])
        prob = np.asarray(values["prob"])
        spikes = np.asarray(values["spikes"])
        n_total = int(values["n_total"])
        max_rate = float(values["max_rate"])
        t = np.arange(len(signal))

        self._ax_top.plot(t, signal, color=SNN_COLOR, linewidth=2, label="sinal (intensidade)")
        self._ax_top.plot(t, prob, color=ACCENT_COLOR, linewidth=1.5, linestyle="--", label="P(spike) por passo")
        if len(t):
            self._ax_top.plot([t[-1]], [signal[-1]], marker="o", markersize=7, color=SNN_COLOR, zorder=4)
            self._ax_top.plot([t[-1]], [prob[-1]], marker="o", markersize=6, color=ACCENT_COLOR, zorder=4)
            self._ax_top.text(
                t[-1], prob[-1] + 0.1, f"P={prob[-1]:.2f}", ha="center", fontsize=8, color=ACCENT_COLOR,
            )
        self._ax_top.axhline(max_rate, color=NEUTRAL_COLOR, linestyle=":", linewidth=1, label=f"taxa máxima = {max_rate:.2f}")
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(-1.1, 1.1)
        self._ax_top.set_ylabel("amplitude / probabilidade")
        self._ax_top.legend(loc="upper right", fontsize=7.5)

        spike_times = t[spikes > 0]
        for st in spike_times:
            self._ax_bottom.text(st, 1.05, f"t={st}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.vlines(spike_times, 0, 1, color=SNN_COLOR, linewidth=2)
        self._ax_bottom.set_yscale("linear")  # backprop_convergence leaves this axis log-scaled otherwise
        self._ax_bottom.set_xlim(0, n_total)
        self._ax_bottom.set_ylim(0, 1)
        self._ax_bottom.set_yticks([])
        self._ax_bottom.set_xlabel("tempo (passos)")
        self._ax_bottom.set_ylabel("spikes sorteados")

    def _render_poisson_image(self, values: dict[str, object]) -> None:
        image = np.asarray(values["image"])
        frame = np.asarray(values["frame"])
        estimate = np.asarray(values.get("estimate", np.zeros_like(image)))
        t = int(values["t"])
        n_total = int(values["n_total"])
        max_rate = float(values["max_rate"])

        # aspect="equal" so pixels stay square (an "auto" aspect stretches
        # the image to fill whatever shape the panel happens to be, which
        # distorts Patrick badly whenever the panel isn't close to the
        # photo's own 16:9 ratio -- exactly what made him unrecognizable).
        # adjustable="datalim" is the part that avoids the *other* bug seen
        # earlier: it keeps the panel's rectangle exactly as positioned
        # below and instead pads the data limits so the correctly-shaped
        # image is centered inside it (blank bars on the sides if the
        # panel is wider than 16:9) -- the default adjustable="box" instead
        # shrinks the panel's own rectangle down to a thin sliver.
        # Pin the view to the whole pixel grid on every render. imshow() does
        # not reset the axes' view, and this widget is one long-lived instance
        # re-rendered as the user switches demos, so the limits left behind by
        # the previous demo (or by the last step) would otherwise stick and
        # show only a zoomed-in corner of Patrick on return (fast_clear only
        # drops artists, not limits). ylim is flipped because imshow's default
        # origin is "upper" (row 0 at the top). adjustable="datalim" then pads
        # these base limits to the panel's shape, keeping pixels square and
        # the whole photo visible every time.
        rows, cols = image.shape
        # Original and the rate-decoded estimate side by side in ONE image,
        # with a blank (NaN -> background) gutter between them: the claim
        # "only the sum of steps rebuilds the picture" is then visible as
        # the right half sharpening toward the left half.
        gutter = max(2, cols // 24)
        pair = np.full((rows, 2 * cols + gutter), np.nan)
        pair[:, :cols] = image
        pair[:, cols + gutter:] = estimate
        pair_cols = pair.shape[1]
        self._ax_top.imshow(pair, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
        self._ax_top.set_aspect("equal", adjustable="datalim")
        self._ax_top.set_xlim(-0.5, pair_cols - 0.5)
        self._ax_top.set_ylim(rows - 0.5, -0.5)
        self._ax_top.set_xticks([])
        self._ax_top.set_yticks([])
        self._ax_top.set_title(
            f"Original (esq.)  |  estimativa = spikes até t ÷ (t+1) ÷ taxa máxima, t = {t} (dir.)"
        )

        self._ax_bottom.imshow(frame, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
        self._ax_bottom.set_aspect("equal", adjustable="datalim")
        self._ax_bottom.set_xlim(-0.5, cols - 0.5)
        self._ax_bottom.set_ylim(rows - 0.5, -0.5)
        self._ax_bottom.set_xticks([])
        self._ax_bottom.set_yticks([])
        self._ax_bottom.set_title(f"Spikes sorteados no passo t={t}/{n_total - 1} (taxa máxima = {max_rate:.2f})")

    def _render_lif(self, values: dict[str, object]) -> None:
        current = np.asarray(values["current"])
        membrane = np.asarray(values["membrane"])
        spikes = np.asarray(values["spikes"])
        v_th = float(values["v_th"])
        t = np.arange(len(membrane))

        self._ax_top.plot(t, current, color=ACCENT_COLOR, linewidth=2)
        if len(t):
            self._ax_top.plot([t[-1]], [current[-1]], marker="o", markersize=6, color=ACCENT_COLOR, zorder=4)
            self._ax_top.text(t[-1], current[-1] + 0.03, f"{current[-1]:.2f}", ha="center", fontsize=8, color=ACCENT_COLOR)
        self._ax_top.set_ylabel("I(t)")
        self._ax_top.set_ylim(-0.05, max(0.5, current.max()  if len(current) else 0.5))

        self._ax_bottom.plot(t, membrane, color=SNN_COLOR, linewidth=2)
        if len(t):
            self._ax_bottom.plot([t[-1]], [membrane[-1]], marker="o", markersize=7, color=SNN_COLOR, zorder=4)
            self._ax_bottom.text(t[-1], membrane[-1] + 0.05, f"{membrane[-1]:.2f}", ha="center", fontsize=8, color=SNN_COLOR)
        self._ax_bottom.axhline(v_th, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label=f"V_th = {v_th:.2f}")
        spike_times = t[spikes > 0]
        if len(spike_times):
            self._ax_bottom.vlines(spike_times, v_th, v_th * 1.25, color=SNN_COLOR, linewidth=2)
            for st in spike_times:
                self._ax_bottom.text(st, v_th, f"t={st}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.set_yscale("linear")  # backprop_convergence leaves this axis log-scaled otherwise
        self._ax_bottom.set_ylim(-0.1, v_th * 1.4)
        self._ax_bottom.set_xlabel("tempo (passos)")
        self._ax_bottom.set_ylabel("V(t)")
        self._ax_bottom.legend(loc="upper left", fontsize=8)

    def _render_backprop_convergence(self, values: dict[str, object]) -> None:
        iterations = np.asarray(values["iterations"])
        y = np.asarray(values["y"])
        loss = np.asarray(values["loss"])
        target = float(values["target"])
        n_total = int(values["n_total"])

        y_min, y_max = float(values["y_min"]), float(values["y_max"])
        self._ax_top.plot(iterations, y, color=CONVERGE_COLOR, linewidth=2, marker="o", markersize=4)
        self._ax_top.plot([iterations[-1]], [y[-1]], marker="o", markersize=9, color=CONVERGE_COLOR, zorder=4)
        self._ax_top.axhline(target, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label=f"alvo = {target:g}")
        # every point gets its value labelled -- only a handful of iterations
        # exist, so this stays readable, unlike the 60-sample traces above.
        label_dy = (y_max - y_min) * 0.05
        for i, (xi, yi) in enumerate(zip(iterations, y)):
            va = "bottom" if i % 2 == 0 else "top"
            dy = label_dy if va == "bottom" else -label_dy
            self._ax_top.text(xi, yi + dy, f"{yi:.2f}", ha="center", va=va, fontsize=7, color=CONVERGE_COLOR)
        self._ax_top.set_xlim(0, n_total)
        self._ax_top.set_ylim(y_min, y_max)
        self._ax_top.set_ylabel("y (saída)")
        self._ax_top.legend(loc="lower right", fontsize=8)

        self._ax_bottom.plot(iterations, loss, color=SNN_COLOR, linewidth=2, marker="o", markersize=4)
        self._ax_bottom.plot([iterations[-1]], [loss[-1]], marker="o", markersize=8, color=SNN_COLOR, zorder=4)
        for xi, li in zip(iterations, loss):
            self._ax_bottom.text(xi, li * 1.3, f"{li:.3f}", ha="center", fontsize=7, color=SNN_COLOR)
        self._ax_bottom.set_yscale("log")
        self._ax_bottom.set_xlim(0, n_total)
        self._ax_bottom.set_ylim(float(values["loss_min"]), float(values["loss_max"]))
        self._ax_bottom.set_xlabel("iteração")
        self._ax_bottom.set_ylabel("loss (log)")

        # point_y is the activation value at the *step-locked* inset z, which
        # during the per-iteration substeps is the previous iteration's z —
        # not the sliding chart tip (y[-1]) — so the dot stays on the curve
        # at a completed step instead of gliding toward the next one.
        self._draw_sigmoid_inset(
            rect=(0.68, 0.13, 0.29, 0.78),
            z=float(values["z"]), y=float(values.get("point_y", y[-1])),
            slope=float(values["slope"]), grad_z=float(values["grad_z"]),
            z_trail=np.asarray(values["z_trail"]),
        )

    def _render_firing_rate_reg(self, values: dict[str, object]) -> None:
        """Two zoomed panels, one per failure: the nearly dead LAYER's mean
        rate climbing to the floor (top) and the bursting layer's mean
        descending to the ceiling (bottom). The arrow at each tip is the
        regularizer's push on that mean, drawn in rate units -- it visibly
        shrinks to nothing at the band's edge. With ``mixed_reveal`` on, a
        dashed line per panel marks the two units of a third, mixed layer
        whose mean sits inside the band and therefore gets no push."""
        epochs = np.asarray(values["epochs"])
        n_total = int(values["n_total"])
        r_min = float(values["r_min"])
        r_max = float(values["r_max"])
        push_reveal = float(values["push_reveal"])
        mixed_reveal = float(values.get("mixed_reveal", 0.0))
        mixed_mean = float(values.get("mixed_mean", 0.0))
        mixed_push = float(values.get("mixed_push", 0.0))
        panels = (
            (self._ax_top, "dead", SNN_COLOR, "camada quase morta", (0.0, float(values["dead_top"])), r_min, "piso r_min"),
            (self._ax_bottom, "burst", BITNET_COLOR, "camada em rajada", (float(values["burst_bottom"]), 1.0), r_max, "teto r_max"),
        )
        x_right = n_total * 1.5  # room for the tip labels
        for ax, key, color, name, (lo, hi), edge, edge_name in panels:
            rates = np.asarray(values[f"rate_{key}"])
            push = float(values[f"push_{key}"])
            loss = float(values[f"loss_{key}"])
            ax.axhspan(r_min, r_max, color=CONVERGE_COLOR, alpha=0.13)
            ax.axhline(edge, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
            ax.text(
                0.3, edge, f"{edge_name} = {edge:.2f}", va="bottom" if key == "dead" else "top",
                fontsize=8, color=NEUTRAL_COLOR,
            )
            ax.plot(epochs, rates, color=color, linewidth=2, marker="o", markersize=3.5)
            if len(epochs):
                tip_x, tip_y = float(epochs[-1]), float(rates[-1])
                ax.plot([tip_x], [tip_y], marker="o", markersize=8, color=color, zorder=4)
                ax.text(
                    tip_x + 0.6, tip_y, f"{name}: média {tip_y:.3f}\nL_reg = {loss:.5f}",
                    va="center", fontsize=8, color=color, weight="bold",
                )
                if push_reveal > 0.02 and abs(push) > 1e-4:
                    ax.annotate(
                        "", xy=(tip_x, tip_y + push), xytext=(tip_x, tip_y),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT_COLOR, linewidth=2.2, alpha=push_reveal),
                    )
            if mixed_reveal > 0.02:
                unit_rate = float(values[f"mixed_{key}"])
                ax.axhline(unit_rate, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.6, alpha=mixed_reveal)
                ax.text(
                    n_total * 0.5, unit_rate,
                    f"camada mista: unidade em {unit_rate:.2f}, média {mixed_mean:.3f}, empurrão {mixed_push:+.3f}",
                    va="bottom", ha="center", fontsize=7.5, color=NEUTRAL_COLOR, alpha=mixed_reveal,
                )
            ax.set_yscale("linear")  # backprop_convergence leaves the bottom axis log-scaled otherwise
            ax.set_ylim(lo, hi)
            ax.set_xlim(0, x_right)
            ax.set_ylabel("taxa média")
        self._ax_top.set_title("Faixa alvo (verde): o regularizador só empurra a MÉDIA de camada que está fora dela", fontsize=9.5)
        self._ax_bottom.set_xlabel("época")

    def _render_encoding_loss_mismatch(self, values: dict[str, object]) -> None:
        """Top: WHERE each unit fires (the thing a latency code carries).
        Bottom: what the training loss REPORTS for that same run. The
        failure shows up as the two panels disagreeing."""
        iterations = np.asarray(values["iterations"])
        n_total = int(values["n_total"])
        time_steps = int(values["time_steps"])
        target_frame = float(values["target_frame"])
        never_reveal = float(values["never_reveal"])
        runs = (
            (np.asarray(values["frame_a"]), np.asarray(values["loss_a"]), values["label_a"], CONVERGE_COLOR),
            (np.asarray(values["frame_b"]), np.asarray(values["loss_b"]), values["label_b"], SNN_COLOR),
        )

        top, bottom = self._ax_top, self._ax_bottom
        top.axhline(target_frame, color=CONVERGE_COLOR, linestyle="--", linewidth=1.2)
        top.text(
            0.2, target_frame - 0.3, f"alvo: quadro {target_frame:g}", ha="left", va="top", fontsize=8,
            color=CONVERGE_COLOR,
        )
        if never_reveal > 0.02:
            top.axhline(time_steps, color=SNN_COLOR, linestyle=":", linewidth=1.4, alpha=never_reveal)
            top.text(
                0.2, time_steps + 0.3, f"t = T = {time_steps}: nunca disparou", ha="left", va="bottom",
                fontsize=8, color=SNN_COLOR, alpha=never_reveal,
            )
        # Labelled at the moving tip rather than in a legend: any legend
        # corner sits on one of the two runs in one of the two acts.
        for index, (frames, losses, label, color) in enumerate(runs):
            top.step(iterations, frames, where="post", color=color, linewidth=2.2)
            bottom.step(iterations, losses, where="post", color=color, linewidth=2.2)
            if not len(iterations):
                continue
            above = index == 1
            top.plot([iterations[-1]], [frames[-1]], marker="o", markersize=8, color=color, zorder=4)
            top.text(
                iterations[-1] + 0.25, frames[-1] + (0.4 if above else -0.4), f"{label}: quadro {frames[-1]:g}",
                ha="left", va="bottom" if above else "top", fontsize=8, color=color, weight="bold",
            )
            bottom.plot([iterations[-1]], [losses[-1]], marker="o", markersize=7, color=color, zorder=4)
            bottom.text(
                iterations[-1] + 0.25, losses[-1], f"{losses[-1]:g}", va="center", fontsize=8.5,
                color=color, weight="bold",
            )
        x_right = n_total * 1.4  # room for the tip labels past the last epoch
        top.set_xlim(0, x_right)
        top.set_ylim(-0.5, time_steps + 1.5)
        top.set_ylabel("quadro do 1º disparo")
        top.set_title(values["act_title"], fontsize=9.5)

        bottom.set_yscale("linear")  # backprop_convergence leaves this axis log-scaled otherwise
        bottom.set_xlim(0, x_right)
        bottom.set_ylim(-0.08 * float(values["loss_max"]), float(values["loss_max"]) * 1.15)
        bottom.set_ylabel("perda reportada")
        bottom.set_xlabel("época")

    def _draw_sigmoid_inset(
        self, rect: tuple[float, float, float, float], z: float, y: float, slope: float, grad_z: float,
        z_trail: np.ndarray | None = None,
    ) -> None:
        # same construction as neuron_view's inset -- kept local (not
        # shared) since the two widgets have no other coupling and this is
        # the only spot signal_view needs it. Cached rather than recreated
        # every frame -- see widgets/_mpl_perf.py.
        if self._inset_ax is None:
            self._inset_ax = self._figure.add_axes(rect)
        else:
            self._inset_ax.set_position(rect)
            fast_clear(self._inset_ax)
            self._inset_ax.set_visible(True)
        ax = self._inset_ax

        z_grid = np.linspace(-6.0, 6.0, 200)
        ax.plot(z_grid, 1.0 / (1.0 + np.exp(-z_grid)), color=CONVERGE_COLOR, linewidth=2)
        ax.axvline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")

        # every past iteration's point stays marked on the curve -- a new
        # dot is added each iteration, exactly like the y-vs-iteration
        # chart to its left, instead of one dot relocating and erasing
        # where it has already been.
        if z_trail is not None and len(z_trail) > 1:
            y_trail = 1.0 / (1.0 + np.exp(-z_trail[:-1]))
            ax.plot(z_trail[:-1], y_trail, marker="o", markersize=5, color=CONVERGE_COLOR, alpha=0.4, linestyle="None", zorder=3)

        ax.plot([z], [y], marker="o", markersize=9, color=CONVERGE_COLOR, zorder=4)
        ax.text(z, y + 0.08, f"y = {y:.2f}", ha="center", fontsize=9, color=CONVERGE_COLOR)

        half = 2.0
        z_tan = np.array([z - half, z + half])
        y_tan = y + slope * (z_tan - z)
        ax.plot(z_tan, y_tan, color=ACCENT_COLOR, linewidth=1.5, linestyle="--")
        ax.text(z_tan[0], y_tan[0], f"σ'(z) = {slope:.2f}", ha="right", va="top", fontsize=8.5, color=ACCENT_COLOR)

        direction = -1.0 if grad_z >= 0 else 1.0
        dz = direction * 0.9
        dy = slope * dz
        # cheap shaft + rotated-triangle arrowhead, not ax.annotate's
        # FancyArrowPatch -- see neuron_view._flow_arrow for why.
        ax.plot([z, z + dz], [y, y + dy], color=SNN_COLOR, linewidth=2)
        angle = degrees(atan2(dy, dz)) - 90.0
        ax.plot([z + dz], [y + dy], marker=(3, 0, angle), markersize=10, color=SNN_COLOR, linestyle="None")
        ax.text(
            z + dz, y + dy + (0.1 if direction > 0 else -0.15),
            "descida do gradiente", ha="center", fontsize=8.5, color=SNN_COLOR,
        )

        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("z", fontsize=8)
        ax.set_ylabel("σ(z)", fontsize=8)
        ax.set_title("Ativação: onde estamos na curva", fontsize=8.5)
        ax.tick_params(labelsize=7)
