"""Renderer for `snn.timesteps`'s `timesteps_tensor` frame kind
(software/nn's .wiki/Concepts/Time-Steps.md).

Two scenes. The movie: one sample as T numbered frames, a playhead sweeping
across them and latency spikes lighting up as it passes; a brace over all
frames names T ("how many") and a bracket under one frame names delta_t
("how long") -- the confusable pair drawn side by side. The tensor: the
same physical (t, b) cells sliding between time-major and batch-major row
order, next to a fixed column saying how LifBPTT reads each row, so the
wrong order shows up as visible mismatches rather than a claim.
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, CONVERGE_COLOR, NEUTRAL_COLOR, SNN_COLOR

_ROW_SAMPLE_COLORS = (BITNET_COLOR, SNN_COLOR, ACCENT_COLOR)


class TimestepsRendererMixin:
    def _render_timesteps_tensor(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        if values["stage"] == "movie":
            self._render_timesteps_movie(values)
        else:
            self._render_timesteps_tensor_grid(values)

    def _render_timesteps_movie(self, values: dict[str, object]) -> None:
        ax = self._ax
        time_steps = int(values["time_steps"])
        playhead = float(values["playhead"])
        dt_reveal = float(values["dt_reveal"])

        box_w = 0.9 / time_steps
        y = 0.52
        current = int(round(playhead))
        for t in range(time_steps):
            x = 0.05 + (t + 0.5) * box_w
            is_now = t == current
            ax.add_patch(
                Rectangle(
                    (x - box_w * 0.4, y - 0.1), box_w * 0.8, 0.2,
                    facecolor=CONVERGE_COLOR if is_now else "none",
                    edgecolor=CONVERGE_COLOR if is_now else NEUTRAL_COLOR,
                    alpha=0.45 if is_now else 0.5, linewidth=1.6 if is_now else 1.0,
                )
            )
            ax.text(x, y - 0.15, str(t), ha="center", va="center", fontsize=7, color=NEUTRAL_COLOR)
        cursor_x = 0.05 + (playhead + 0.5) * box_w
        ax.plot([cursor_x, cursor_x], [y - 0.12, y + 0.12], color=CONVERGE_COLOR, linewidth=2.2, zorder=6)

        for frame_idx, feature_value, color in values["example_spikes"]:
            frame_idx = int(round(frame_idx))
            if playhead + 1e-6 < frame_idx:
                continue  # o cursor ainda não chegou neste quadro
            x = 0.05 + (frame_idx + 0.5) * box_w
            ax.plot([x], [y], marker="o", markersize=11, color=color, zorder=5)
            ax.text(
                x, y + 0.13, f"x = {feature_value:.1f}\n→ quadro {frame_idx}",
                ha="center", va="bottom", fontsize=7.5, color=color,
            )

        # time_steps: a brace over ALL frames -- how MANY.
        brace_y = 0.8
        ax.plot([0.05, 0.95], [brace_y, brace_y], color=NEUTRAL_COLOR, linewidth=1.2)
        for edge in (0.05, 0.95):
            ax.plot([edge, edge], [brace_y - 0.03, brace_y], color=NEUTRAL_COLOR, linewidth=1.2)
        ax.text(
            0.5, brace_y + 0.03, f"time_steps = T = {time_steps} quadros  (QUANTOS)",
            ha="center", va="bottom", fontsize=9.5, weight="bold",
        )

        # delta_t: a bracket under ONE frame -- how LONG.
        if dt_reveal > 0.02:
            x0 = 0.05 + 0.1 * box_w
            x1 = 0.05 + 0.9 * box_w
            bracket_y = 0.27
            ax.plot([x0, x1], [bracket_y, bracket_y], color=ACCENT_COLOR, linewidth=2, alpha=dt_reveal)
            for edge in (x0, x1):
                ax.plot([edge, edge], [bracket_y, bracket_y + 0.03], color=ACCENT_COLOR, linewidth=2, alpha=dt_reveal)
            ax.text(
                x0, bracket_y - 0.03, "delta_t = Δt  (QUANTO DURA um quadro)",
                ha="left", va="top", fontsize=9, color=ACCENT_COLOR, alpha=dt_reveal, weight="bold",
            )

        ax.text(
            0.5, 0.96, f"quadro atual: {current}", ha="center", va="top", fontsize=9, color=CONVERGE_COLOR,
        )
        ax.text(
            0.5, 0.06, "Valor alto dispara cedo; valor baixo, tarde. Cada feature dispara exatamente uma vez.",
            ha="center", fontsize=8.5, color=NEUTRAL_COLOR,
        )
        ax.axis("off")

    def _render_timesteps_tensor_grid(self, values: dict[str, object]) -> None:
        ax = self._ax
        total_rows = int(values["total_rows"])
        n_samples = int(values["n_samples"])
        row_y = values["row_y"]
        row_t = values["row_t"]
        row_sample = values["row_sample"]
        slot_y = values["slot_y"]
        cell_h = 0.72 / total_rows

        ax.text(0.5, 0.99, values["order_title"], ha="center", va="top", fontsize=9.5, weight="bold")
        ax.text(0.27, 0.9, "linha", ha="right", va="bottom", fontsize=8, color=NEUTRAL_COLOR)
        ax.text(0.45, 0.9, "conteúdo real", ha="center", va="bottom", fontsize=8, color=NEUTRAL_COLOR)
        ax.text(0.62, 0.9, "LifBPTT lê como", ha="left", va="bottom", fontsize=8, color=NEUTRAL_COLOR)
        for r, sy in enumerate(slot_y):
            ax.text(0.27, sy, str(r), ha="right", va="center", fontsize=8, color=NEUTRAL_COLOR)

        settled: dict[int, tuple[int, int]] = {}
        for i in range(total_rows):
            y = float(row_y[i])
            t = int(round(row_t[i]))
            b = int(round(row_sample[i]))
            ax.add_patch(
                Rectangle(
                    (0.35, y - cell_h * 0.42), 0.2, cell_h * 0.84,
                    facecolor=_ROW_SAMPLE_COLORS[b], alpha=0.85, edgecolor="none",
                )
            )
            ax.text(0.45, y, f"t={t}, b={b}", ha="center", va="center", fontsize=7.5, color="white", weight="bold")
            for r, sy in enumerate(slot_y):
                if abs(y - float(sy)) < cell_h * 0.05:
                    settled[r] = (t, b)

        # LifBPTT never sees the cells' labels: it infers (t, b) from the row
        # index alone (t = r // B, b = r % B). Marked only once a cell has
        # settled in a slot, so nothing is judged mid-slide.
        for r, sy in enumerate(slot_y):
            expected = (r // n_samples, r % n_samples)
            actual = settled.get(r)
            if actual is None:
                color, suffix = NEUTRAL_COLOR, ""
            elif actual == expected:
                color, suffix = CONVERGE_COLOR, ""
            else:
                color, suffix = SNN_COLOR, "  ← errado"
            ax.text(
                0.62, sy, f"t={expected[0]}, b={expected[1]}{suffix}",
                ha="left", va="center", fontsize=7.5, color=color, weight="bold" if suffix else "normal",
            )

        for b in range(n_samples):
            x = 0.3 + 0.17 * b
            ax.add_patch(Rectangle((x, 0.02), 0.03, 0.035, facecolor=_ROW_SAMPLE_COLORS[b], edgecolor="none"))
            ax.text(x + 0.04, 0.037, f"amostra b={b}", ha="left", va="center", fontsize=8, color=NEUTRAL_COLOR)
        ax.axis("off")
