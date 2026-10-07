"""Renders single-weight / single-function persistent scenes: a number
line for the scalar quantization demo, the quantization staircase (with a
fading-in annotation), and the SNN surrogate-gradient curve (which morphs
continuously from the useless true derivative into the smooth surrogate).

Each ``render(values)`` call redraws the *same* picture for a given demo,
just with different continuous field values — routing is by the frame's
``kind`` tag, which stays constant across a whole demo, so this widget
never has to guess which layout to build.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from efficient_nn_lab.app.theme import (
    ACCENT_COLOR,
    BITNET_COLOR,
    NEUTRAL_COLOR,
    PARACONSISTENT_COLOR,
    SNN_COLOR,
    TEXT_COLOR,
)
from efficient_nn_lab.paraconsistent.ga_synthetic import constraint_violation, first_front_mask
from efficient_nn_lab.paraconsistent.metrics import VERTICES, d_penalized
from efficient_nn_lab.widgets._mpl_perf import fast_clear


class WeightView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 3.2))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def render(self, values: dict[str, object]) -> None:
        fast_clear(self._ax)
        kind = values.get("kind")
        if kind == "scalar_quantization":
            self._render_number_line(values)
        elif kind == "staircase":
            self._render_staircase(values)
        elif kind == "quant_derivative":
            self._render_quant_derivative(values)
        elif kind == "surrogate_curve":
            self._render_surrogate_curve(values)
        elif kind in ("paraconsistent_plane", "paraconsistent_dpenalized"):
            self._render_paraconsistent_plane(values)
        elif kind == "paraconsistent_ga_pareto":
            self._render_paraconsistent_ga_pareto(values)
        elif kind == "encoding_noise_floor":
            self._render_encoding_noise_floor(values)
        elif kind == "tdbn_distribution":
            self._render_tdbn_distribution(values)
        else:
            self._ax.text(0.5, 0.5, "(sem visualização para este passo)", ha="center", va="center")
            self._ax.axis("off")
        self._canvas.draw_idle()

    # -- scalar_quantization ------------------------------------------
    def _render_number_line(self, values: dict[str, object]) -> None:
        ax = self._ax
        threshold = float(values["threshold"])
        w_display = float(values["w_display"])
        w_quant = values.get("w_quant")

        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1.5, zorder=1)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.18, zorder=0)
        for level in (-1, 0, 1):
            ax.plot([level], [0], marker="|", markersize=24, color=NEUTRAL_COLOR, zorder=2)
            ax.text(level, -0.28, f"{level:+d}", ha="center", fontsize=11)

        marker_color = ACCENT_COLOR if w_quant is not None else BITNET_COLOR
        ax.plot([w_display], [0], marker="o", markersize=16, color=marker_color, zorder=3)
        ax.text(w_display, 0.22, f"{w_display:.2f}", ha="center", fontsize=11, color=marker_color)
        ax.text(threshold, -0.5, f"τ = {threshold:.2f}", ha="left", fontsize=8, color=NEUTRAL_COLOR, style="italic")

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("valor do peso")
        ax.set_title("O peso desliza até o nível quantizado mais próximo")

    # -- backward staircase ---------------------------------------------
    def _render_staircase(self, values: dict[str, object]) -> None:
        ax = self._ax
        w = values["w"]
        q = values["q"]
        threshold = float(values["threshold"])
        annotate = float(values.get("annotate_reveal", 0.0))

        ax.plot(w, q, color=BITNET_COLOR, linewidth=2.5)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.18)
        ax.axvline(threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.axvline(-threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.text(threshold, -1.35, f"τ = {threshold:.2f}", ha="left", fontsize=8, color=NEUTRAL_COLOR, style="italic")
        ax.text(-threshold, -1.35, f"-τ = {-threshold:.2f}", ha="right", fontsize=8, color=NEUTRAL_COLOR, style="italic")
        for level in (-1, 0, 1):
            ax.text(w.min() + 0.05, level, f"Q(w) = {level:+d}", ha="left", va="bottom", fontsize=8, color=BITNET_COLOR)
        ax.set_xlabel("w (peso real)")
        ax.set_ylabel("Q(w)")
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1.5, 1.2)
        ax.set_title("Regiões planas: derivada zero. Saltos: derivada indefinida.")

        # mark the demo's concrete worked example on the same staircase,
        # so every scene of this demo points at the same w (see backward.py).
        example_w = float(values.get("example_w", np.nan))
        if np.isfinite(example_w):
            q_at = float(q[int(np.argmin(np.abs(w - example_w)))])
            ax.axvline(example_w, color=ACCENT_COLOR, linestyle=":", linewidth=1.2, zorder=4)
            ax.plot([example_w], [q_at], marker="o", markersize=7, color=ACCENT_COLOR, zorder=5)
            ax.text(
                example_w, 1.05, f"w = {example_w:.2f} -> Q(w) = {q_at:+.0f}",
                ha="center", fontsize=8, color=ACCENT_COLOR,
            )

        if annotate > 0.01:
            ax.annotate(
                "derivada = 0",
                xy=(0.0, 0.0),
                xytext=(0.0, 0.55),
                ha="center",
                fontsize=9,
                color=SNN_COLOR,
                alpha=annotate,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR, alpha=annotate),
            )
            ax.annotate(
                "derivada indefinida",
                xy=(threshold, 0.5),
                xytext=(threshold + 0.35, 0.9),
                ha="left",
                fontsize=9,
                color=SNN_COLOR,
                alpha=annotate,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR, alpha=annotate),
            )

    # -- backward derivative graph (real dQ/dw vs. what STE substitutes) --
    def _render_quant_derivative(self, values: dict[str, object]) -> None:
        ax = self._ax
        w = values["w"]
        curve = values["curve"]
        threshold = float(values["threshold"])
        overlay_reveal = float(values.get("overlay_reveal", 0.0))

        ax.plot(w, curve, color=SNN_COLOR, linewidth=2.5, zorder=3)
        ax.axvline(threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.axvline(-threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.axvspan(-threshold, threshold, color=NEUTRAL_COLOR, alpha=0.12, zorder=0)
        # the "current point" is the demo's concrete example weight (see
        # backward.py), not an arbitrary midpoint of the sampled curve, so
        # this scene references the same w as the staircase and the block
        # diagram.
        example_w = float(values.get("example_w", w[len(w) // 2]))
        idx = int(np.argmin(np.abs(np.asarray(w) - example_w)))
        current_level = float(np.asarray(curve)[idx])
        ax.axvline(example_w, color=ACCENT_COLOR, linestyle=":", linewidth=1.2, zorder=4)
        ax.plot([example_w], [current_level], marker="o", markersize=7, color=ACCENT_COLOR, zorder=5)
        ax.text(
            w.min() + 0.1, current_level + 0.12,
            f"dQ/dw({example_w:.2f}) = {current_level:.2f}", ha="left", fontsize=9, color=SNN_COLOR,
        )

        if overlay_reveal <= 0.02:
            ax.annotate(
                "indefinida aqui",
                xy=(threshold, 0.0),
                xytext=(threshold + 0.25, 0.55),
                ha="left",
                fontsize=9,
                color=SNN_COLOR,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR),
            )
            ax.annotate(
                "indefinida aqui",
                xy=(-threshold, 0.0),
                xytext=(-threshold - 0.25, 0.55),
                ha="right",
                fontsize=9,
                color=SNN_COLOR,
                arrowprops=dict(arrowstyle="->", color=SNN_COLOR),
            )
            ax.set_title("A derivada real de Q(w): zero em toda parte plana")
        else:
            ax.plot(
                w, np.zeros_like(w), color=NEUTRAL_COLOR, linewidth=1.5, linestyle="--",
                alpha=overlay_reveal, label="derivada real (para comparação)", zorder=2,
            )
            ax.legend(loc="center left", fontsize=8)
            ax.set_title("O STE troca essa derivada por uma constante 1 (identidade)")

        ax.set_xlabel("w (peso real)")
        ax.set_ylabel("dQ/dw usada")
        ax.set_xlim(w.min(), w.max())
        ax.set_ylim(-0.3, 1.5)

    # -- surrogate gradient -----------------------------------------------
    def _render_surrogate_curve(self, values: dict[str, object]) -> None:
        ax = self._ax
        x = values["x"]
        bottom_reveal = float(values.get("bottom_reveal", 0.0))
        overlay_reveal = float(values.get("overlay_reveal", 0.0))
        sigmoid_reveal = float(values.get("sigmoid_reveal", 0.0))
        draw_reveal = float(values.get("draw_reveal", 0.0))

        example_vmt = float(values.get("example_vmt", 0.2))
        example_v = float(values.get("example_v", 0.0))
        example_spike = float(values.get("example_spike", 1.0))
        example_sigmoid = float(values.get("example_sigmoid", 0.0))
        example_surrogate = float(values.get("example_surrogate", 0.0))

        ax.axvline(0, color=NEUTRAL_COLOR, linewidth=1, linestyle=":")

        if bottom_reveal < 0.02:
            spike = values["spike"]
            ax.plot(x, spike, color=SNN_COLOR, linewidth=2.5, label="S(v) — degrau (forward)")
            ax.text(0.05, 1.05, "S(0) = 1", ha="left", fontsize=9, color=SNN_COLOR)
            ax.text(-0.35, -0.1, "S(v<0) = 0", ha="right", fontsize=9, color=SNN_COLOR)
            ax.axvline(example_vmt, color=BITNET_COLOR, linewidth=1, linestyle=":")
            ax.plot([example_vmt], [example_spike], marker="o", markersize=7, color=BITNET_COLOR, zorder=6)
            if sigmoid_reveal > 0.02:
                sigmoid = values["sigmoid"]
                ax.plot(
                    x, sigmoid, color=ACCENT_COLOR, linewidth=2, linestyle="--", alpha=sigmoid_reveal,
                    label="sigmoide suave (antiderivada do gradiente substituto)",
                )
                ax.plot([example_vmt], [example_sigmoid], marker="o", markersize=7, color=BITNET_COLOR, zorder=6)
                mid = len(x) // 2
                ax.text(
                    x[mid], float(sigmoid[mid]) + 0.06, f"sigmoide({x[mid]:.1f}) = {float(sigmoid[mid]):.2f}",
                    ha="center", fontsize=8.5, color=ACCENT_COLOR, alpha=sigmoid_reveal,
                )
                ax.legend(loc="lower right", fontsize=7)
            ax.set_ylabel("S(v) — spike (forward)")
            ax.set_title("Função de disparo real, usada no forward")
            ax.set_ylim(-0.2, 1.2)
        elif draw_reveal < 0.02:
            true_derivative = values["true_derivative"]
            ax.plot(x, true_derivative, color=SNN_COLOR, linewidth=2.5, label="gradiente (backward)")
            ax.axvline(example_vmt, color=BITNET_COLOR, linewidth=1, linestyle=":")
            ax.plot([example_vmt], [0.0], marker="o", markersize=7, color=BITNET_COLOR, zorder=6)
            ax.text(
                example_vmt, 0.14, f"v = {example_v:g}: dS/dv = 0",
                ha="left", fontsize=8.5, color=BITNET_COLOR,
            )
            ax.set_ylabel("gradiente usado no backward")
            ax.set_title("A derivada real: zero em quase todo ponto — inútil para o backward")
        else:
            # the gradient and the sigmoid it comes from are traced together,
            # left to right, up to the same x -- not faded in all at once --
            # so the height of one at the sweep's leading edge is visibly
            # the slope of the other at that same point.
            surrogate = values["surrogate"]
            sigmoid = values["sigmoid"]
            cut = min(len(x), max(2, int(round(draw_reveal * len(x)))))
            xs = x[:cut]
            ax.plot(xs, sigmoid[:cut], color=ACCENT_COLOR, linewidth=2, linestyle="--", label="sigmoide suave")
            ax.plot(xs, surrogate[:cut], color=SNN_COLOR, linewidth=2.5, label="gradiente (backward)")

            tip = cut - 1
            ax.plot([x[tip], x[tip]], [sigmoid[tip], surrogate[tip]], color=NEUTRAL_COLOR, linewidth=1, linestyle=":")
            ax.plot([x[tip]], [sigmoid[tip]], marker="o", markersize=6, color=ACCENT_COLOR, zorder=5)
            ax.plot([x[tip]], [surrogate[tip]], marker="o", markersize=7, color=SNN_COLOR, zorder=5)
            tip_ha = "right" if x[tip] > x[-1] - 0.5 else "center"
            ax.text(
                x[tip], min(1.1, surrogate[tip] + 0.1),
                f"inclinação da sigmoide aqui = altura do gradiente = {surrogate[tip]:.2f}",
                ha=tip_ha, fontsize=8, color=SNN_COLOR,
            )

            example_idx = int(np.searchsorted(x, example_vmt))
            if cut > example_idx:
                ax.axvline(example_vmt, color=BITNET_COLOR, linewidth=1, linestyle=":")
                ax.plot([example_vmt], [example_surrogate], marker="o", markersize=7, color=BITNET_COLOR, zorder=6)
                ex_ha = "right" if example_vmt > x[-1] - 0.5 else "left"
                ax.text(
                    example_vmt + (0.06 if ex_ha == "left" else -0.06),
                    example_surrogate + 0.08,
                    f"v = {example_v:g}: grad = {example_surrogate:.2f}",
                    ha=ex_ha, fontsize=8.5, color=BITNET_COLOR,
                )

            peak_idx = int(np.argmax(surrogate))
            if cut > peak_idx and surrogate[peak_idx] > 0.05:
                ax.annotate(
                    f"pico = {surrogate[peak_idx]:.2f}",
                    xy=(x[peak_idx], surrogate[peak_idx]),
                    xytext=(x[peak_idx] + 0.35, surrogate[peak_idx] + 0.15),
                    fontsize=8, color=SNN_COLOR,
                    arrowprops=dict(arrowstyle="->", color=SNN_COLOR),
                )
            if overlay_reveal > 0.02:
                ax.plot(
                    x,
                    values["spike"],
                    color=NEUTRAL_COLOR,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=overlay_reveal,
                    label="spike (forward, para comparação)",
                )
            ax.legend(loc="upper left", fontsize=7)
            ax.set_ylabel("gradiente usado no backward")
            ax.set_title("O gradiente nasce da inclinação da sigmoide, ponto a ponto")
        if bottom_reveal >= 0.02:
            ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("v - v_th")

    # -- paraconsistent.plane / paraconsistent.dpenalized ------------------
    def _render_paraconsistent_plane(self, values: dict[str, object]) -> None:
        """The plane with up to three extractors on it: the main point
        (g1, ...) plus optional ``weak_*`` and ``dead_*`` points, each a
        full set of four numeric keys so they glide between checkpoints.
        With the optional ``rank_truth``/``rank_penalized`` reveals, the
        footer ranks the points live by each metric; without them it shows
        the main point's D_truth."""
        ax = self._ax
        distance_reveal = float(values.get("distance_reveal", 1.0))
        vertex_scores = float(values.get("vertex_scores", 0.0))

        for name, (vg1, vg2) in VERTICES.items():
            ax.plot([vg1], [vg2], marker="s", markersize=9, color=NEUTRAL_COLOR, zorder=2)
            dx = 0.1 if vg1 >= 0 else -0.1
            ha = "left" if vg1 >= 0 else "right"
            ax.text(vg1 + dx, vg2, name, fontsize=9, color=NEUTRAL_COLOR, ha=ha, va="center")
            if vertex_scores > 0.02:
                ax.text(
                    vg1 + dx, vg2 - 0.11, f"D_penalized = {d_penalized(vg1, vg2):.2f}", fontsize=8,
                    color=PARACONSISTENT_COLOR, ha=ha, va="center", alpha=vertex_scores, weight="bold",
                )
        diamond = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)]
        ax.plot([p[0] for p in diamond], [p[1] for p in diamond], color=NEUTRAL_COLOR, linewidth=0.8, alpha=0.4)
        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")
        ax.axvline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")

        points = [("", PARACONSISTENT_COLOR, "o", values.get("main_label", ""))]
        if values.get("weak_g1") is not None:
            points.append(("weak_", BITNET_COLOR, "D", "fraco"))
        if values.get("dead_g1") is not None:
            points.append(("dead_", SNN_COLOR, "X", "morto"))
        scored = []
        for prefix, color, marker, name in points:
            # an extractor not measured yet is not on the plane at all; it
            # fades in while gliding out of the shared placeholder.
            alpha = float(values.get(f"{prefix}opacity", 1.0))
            if alpha < 0.02:
                continue
            g1 = float(values[f"{prefix}g1"])
            g2 = float(values[f"{prefix}g2"])
            d_t = float(values[f"{prefix}d_truth"])
            d_p = float(values[f"{prefix}d_penalized"])
            if alpha > 0.5:
                scored.append((name, d_t, d_p, color))
            if distance_reveal > 0.02:
                ax.plot(
                    [g1, 1.0], [g2, 0.0], color=color, linewidth=1.2, linestyle="--",
                    alpha=0.7 * distance_reveal * alpha, zorder=3,
                )
            ax.plot([g1], [g2], marker=marker, markersize=13, color=color, alpha=alpha, zorder=5)
            text = f"{name}  " if name else ""
            ax.text(
                g1, g2 + 0.12, f"{text}(G1 = {g1:.3f}, G2 = {g2:.3f})", ha="center", va="bottom",
                fontsize=8.5, color=color, weight="bold", alpha=alpha,
            )

        rank_truth = float(values.get("rank_truth", 0.0))
        rank_penalized = float(values.get("rank_penalized", 0.0))
        if "rank_truth" in values:
            def ranking(metric: int) -> str:
                ordered = sorted(scored, key=lambda s: s[metric])
                return "  <  ".join(f"{s[0]} {s[metric]:.3f}" for s in ordered)
            if rank_truth > 0.02:
                ax.text(-1.45, -1.25, f"D_truth:       {ranking(1)}", ha="left", fontsize=8.5, alpha=rank_truth)
            if rank_penalized > 0.02:
                ax.text(
                    -1.45, -1.42, f"D_penalized: {ranking(2)}", ha="left", fontsize=8.5,
                    color=PARACONSISTENT_COLOR, alpha=rank_penalized, weight="bold",
                )
        elif distance_reveal > 0.02 and scored:
            ax.text(
                0.0, -1.35, f"D_truth = distância até Verdade = {scored[0][1]:.4f}", ha="center",
                fontsize=9, color=PARACONSISTENT_COLOR, alpha=distance_reveal, weight="bold",
            )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.3)
        ax.set_xlabel("G1 = α − β  (certeza)")
        ax.set_ylabel("G2 = α + β − 1  (contradição)")
        ax.set_title("Plano paraconsistente")

    # -- paraconsistent.ga_pareto ------------------------------------------
    def _render_paraconsistent_ga_pareto(self, values: dict[str, object]) -> None:
        """Cost x D_penalized scatter of one (possibly mid-tween) population.

        Feasibility and the front are recomputed here from the frame's own
        positions and ceiling -- never carried in the frame -- so the front
        genuinely redraws while points glide between generations and while
        the ceiling slides past them. Everything is labeled on the plot
        itself: the explanation panel is optional, the picture is not.
        """
        ax = self._ax
        cost = np.asarray(values["cost"], dtype=float)
        d = np.asarray(values["d"], dtype=float)
        x_max, y_max = float(values["x_max"]), 2.1
        ceiling_x = float(values["ceiling_x"])
        ceiling_reveal = float(values["ceiling_reveal"])
        front_reveal = float(values["front_reveal"])
        marks_reveal = float(values["marks_reveal"])
        box_reveal = float(values["box_reveal"])
        menu_reveal = float(values["menu_reveal"])

        violation = constraint_violation(cost, ceiling_x)
        infeasible = violation > 0
        front = first_front_mask(cost, d, violation)

        if ceiling_reveal > 0.02:
            ax.axvspan(ceiling_x, x_max, color=NEUTRAL_COLOR, alpha=0.12 * ceiling_reveal, zorder=0)
            ax.axvline(ceiling_x, color=SNN_COLOR, linestyle="--", linewidth=1.4, alpha=ceiling_reveal, zorder=1)
            # The band under the ideal trade-off curve is empty in every
            # generation, so labels at the bottom never cover a point.
            ax.text(
                ceiling_x - 0.02, 0.06, f"teto de latência = {float(values['ceiling_value']):.2f}",
                ha="right", va="bottom", fontsize=8, color=SNN_COLOR, alpha=ceiling_reveal,
            )
            if x_max - ceiling_x > 0.22:
                ax.text(
                    (ceiling_x + x_max) / 2, 0.06, "inviável", ha="center", va="bottom",
                    fontsize=8, color=NEUTRAL_COLOR, alpha=ceiling_reveal, weight="bold",
                )

        if box_reveal > 0.02:
            a = int(round(values["focus"][0]))
            ca, da = cost[a], d[a]
            ax.fill_between([ca, x_max], da, y_max, color=PARACONSISTENT_COLOR, alpha=0.13 * box_reveal, zorder=0)
            ax.text(
                ca + 0.03, y_max - 0.08, "A domina tudo aqui:\nmais caro E pior que A", ha="left", va="top",
                fontsize=8, color=PARACONSISTENT_COLOR, alpha=box_reveal, weight="bold",
            )
            ax.plot([0, ca, ca], [da, da, 0], color=NEUTRAL_COLOR, linestyle=":", linewidth=1.2, alpha=box_reveal)
            ax.text(
                0.02, da / 2, "região vazia:\nninguém domina A", ha="left", va="center",
                fontsize=7.5, color=NEUTRAL_COLOR, alpha=box_reveal,
            )

        plain = ~front & ~infeasible
        # Before the front step nothing has been called "dominated" yet --
        # and the gray dots then include A and C, which are not.
        plain_label = "dominado" if front_reveal > 0.5 else "arquitetura (treinada e pontuada)"
        ax.scatter(cost[plain], d[plain], s=36, color=NEUTRAL_COLOR, alpha=0.75, zorder=3, label=plain_label)
        over = ~front & infeasible
        if over.any():
            ax.scatter(
                cost[over], d[over], s=44, marker="x", color=NEUTRAL_COLOR, zorder=3,
                label="inviável (custo > teto)",
            )
        fx, fy = cost[front], d[front]
        order = np.argsort(fx)
        if front_reveal < 0.98:  # until revealed, front members look like everyone else
            ax.scatter(fx, fy, s=36, color=NEUTRAL_COLOR, alpha=0.75 * (1.0 - front_reveal), zorder=3)
        if front_reveal > 0.02:
            ax.plot(
                fx[order], fy[order], color=PARACONSISTENT_COLOR, linewidth=2, marker="o", markersize=7,
                alpha=front_reveal, zorder=4, label="fronteira de Pareto",
            )

        if marks_reveal > 0.02:
            for name, index in zip("ABC", values["focus"]):
                i = int(round(index))
                ax.scatter(
                    [cost[i]], [d[i]], s=190, facecolors="none", edgecolors=TEXT_COLOR, linewidths=1.6,
                    alpha=marks_reveal, zorder=6,
                )
                ax.annotate(
                    name, (cost[i], d[i]), xytext=(8, 8), textcoords="offset points", ha="left",
                    fontsize=10, weight="bold", color=TEXT_COLOR, alpha=marks_reveal, zorder=6,
                )

        if menu_reveal > 0.02:
            lo, hi = (int(round(i)) for i in values["menu"])
            for i, text, dy in ((lo, "mais barata", 10), (hi, "melhor D", 22)):
                right_side = cost[i] < 0.65 * x_max
                ax.annotate(
                    f"{text}\ncusto {cost[i]:.2f}, D {d[i]:.2f}", (cost[i], d[i]),
                    xytext=(12 if right_side else -12, dy), textcoords="offset points",
                    ha="left" if right_side else "right", fontsize=8, weight="bold",
                    color=PARACONSISTENT_COLOR, alpha=menu_reveal, zorder=6,
                    arrowprops=dict(arrowstyle="->", color=PARACONSISTENT_COLOR, alpha=menu_reveal),
                )

        generation = int(np.floor(float(values["generation"]) + 1e-6)) + 1
        ax.set_title(f"Geração {generation}/{int(values['n_generations'])} — NSGA-II, dados sintéticos")
        ax.set_xlabel("custo de inferência, normalizado  (← mais barato)")
        ax.set_ylabel("D_penalized  (↓ melhor)")
        ax.legend(loc="upper right", fontsize=7.5)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

    # -- snn.encoding_noise -------------------------------------------------
    def _render_encoding_noise_floor(self, values: dict[str, object]) -> None:
        ax = self._ax
        t_range = np.asarray(values["t_range"])
        poisson_sigma = np.asarray(values["poisson_sigma"])
        latency_error = np.asarray(values["latency_error"])
        t_current = float(values["t_current"])
        poisson_at_t = float(values["poisson_at_t"])
        latency_at_t = float(values["latency_at_t"])
        # 0..1 emphasis per encoding, tweened between checkpoints so the
        # spotlight glides from one curve to the next instead of snapping.
        w_p = float(values["w_poisson"])
        w_l = float(values["w_latency"])
        w_d = float(values["w_direct"])

        def alpha(w: float) -> float:
            return 0.2 + 0.8 * w

        ax.plot(
            t_range, poisson_sigma, color=SNN_COLOR, linewidth=1.4 + 1.8 * w_p, alpha=alpha(w_p),
            label="Poisson: σ do estimador, pior caso p = 0.5  (0.5/√T)",
        )
        ax.plot(
            t_range, latency_error, color=BITNET_COLOR, linewidth=1.4 + 1.8 * w_l, alpha=alpha(w_l),
            label="latência: erro máximo de quantização  (0.5/(T−1))",
        )
        ax.axhline(
            0.0, color=NEUTRAL_COLOR, linewidth=1.0 + 2.2 * w_d, alpha=alpha(w_d),
            label="direta: sem ruído estrutural  (0)",
        )

        ax.plot([t_current], [poisson_at_t], marker="o", markersize=9, color=SNN_COLOR, alpha=alpha(w_p), zorder=4)
        ax.text(
            t_current, poisson_at_t + 0.012, f"{poisson_at_t:.3f}", ha="center", fontsize=8.5,
            color=SNN_COLOR, alpha=alpha(w_p),
        )
        ax.plot([t_current], [latency_at_t], marker="o", markersize=9, color=BITNET_COLOR, alpha=alpha(w_l), zorder=4)
        ax.text(
            t_current, latency_at_t + 0.012, f"{latency_at_t:.4f}", ha="center", fontsize=8.5,
            color=BITNET_COLOR, alpha=alpha(w_l),
        )
        ax.axvline(t_current, color=NEUTRAL_COLOR, linewidth=1, linestyle=":")
        y_top = max(float(poisson_sigma.max()), float(latency_error.max())) * 1.15
        on_left = t_current < (float(t_range.min()) + float(t_range.max())) / 2.0
        ax.text(
            t_current + (0.8 if on_left else -0.8), y_top * 0.55, f"T = {int(round(t_current))}",
            ha="left" if on_left else "right", va="center", fontsize=9.5, color=NEUTRAL_COLOR, weight="bold",
        )

        ax.set_xlabel("T (time_steps)")
        ax.set_ylabel("ruído estrutural (antes de qualquer aprendizado)")
        ax.set_title("Por que \"direto > latência > Poisson\" pode ser só o chão de ruído")
        ax.legend(loc="upper right", fontsize=7.5)
        ax.set_xlim(t_range.min(), t_range.max())
        ax.set_ylim(-0.02, max(float(poisson_sigma.max()), float(latency_error.max())) * 1.15)

    # -- snn.tdbn ------------------------------------------------------------
    def _render_tdbn_distribution(self, values: dict[str, object]) -> None:
        """Raw currents as reference ticks; the current stage's values as
        dots that fill in when they pass the firing threshold (a LIF fires
        only ABOVE +V_th, so there is no -V_th line); optionally a ghost row
        with plain BatchNorm's values for comparison. The x-range is fixed
        for the whole demo so the dots move, not the axis."""
        ax = self._ax
        x = np.asarray(values["x"])
        y = np.asarray(values["y"])
        v_th = float(values["v_th"])
        span = float(values["span"])
        ghost_reveal = float(values["ghost_reveal"])

        ax.axvline(v_th, color=ACCENT_COLOR, linestyle="--", linewidth=1.6, zorder=1)
        ax.text(
            v_th, 0.47, f" V_th = {v_th:.2f}  (dispara →)", ha="left", va="top",
            fontsize=8.5, color=ACCENT_COLOR, weight="bold",
        )
        ax.axhline(0.15, color=NEUTRAL_COLOR, linewidth=0.8, alpha=0.5, zorder=0)
        for xi in x:
            ax.plot([xi], [0.15], marker="|", markersize=18, color=NEUTRAL_COLOR, zorder=2)
        ax.text(
            -span * 0.97, 0.22, f"X bruto: {', '.join(f'{v:g}' for v in x)}   (μ = {x.mean():g}, σ² = {x.var():g})",
            ha="left", va="bottom", fontsize=8.5, color=NEUTRAL_COLOR,
        )

        def row(ys: np.ndarray, row_y: float, color: str, label: str, alpha: float) -> None:
            ax.axhline(row_y, color=NEUTRAL_COLOR, linewidth=0.8, alpha=0.5 * alpha, zorder=0)
            fired = 0
            for yi in ys:
                fires = yi > v_th
                fired += int(fires)
                ax.plot(
                    [yi], [row_y], marker="o", markersize=12, zorder=3, alpha=alpha,
                    markerfacecolor=color if fires else "white", markeredgecolor=color, markeredgewidth=2,
                )
                ax.text(yi, row_y - 0.09, f"{yi:.2f}", ha="center", va="top", fontsize=7.5, color=color, alpha=alpha)
            verb = "dispara" if fired == 1 else "disparam"
            ax.text(
                -span * 0.97, row_y + 0.06, f"{label}: {fired} de {len(ys)} {verb}",
                ha="left", va="bottom", fontsize=8.5, color=color, alpha=alpha, weight="bold",
            )

        row(y, -0.12, SNN_COLOR, values["main_label"], 1.0)
        if ghost_reveal > 0.02:
            row(np.asarray(values["ghost_y"]), -0.47, NEUTRAL_COLOR, "BatchNorm comum (cego a V_th)", ghost_reveal)

        ax.set_xlim(-span, span)
        ax.set_ylim(-0.72, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("corrente de entrada do LIF (unidades de tensão)")
        ax.set_title(values["stage_title"])
