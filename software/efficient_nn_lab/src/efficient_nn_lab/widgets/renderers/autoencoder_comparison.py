"""Renderer for `comparison.autoencoders`'s `autoencoder_comparison_pipeline`
frame kind. A standalone mixin, NOT an extension of `ComparisonRendererMixin`
-- that one's "Saída" row and footnote prose hardcode exactly 3 named
architectures (ANN/BitNet/SNN) with different-shaped values per column, not
a loop over a column list, so it cannot safely generalize to this demo.

Layout, in figure coordinates (the main axes span the whole figure, so its
data coordinates ARE figure coordinates): the test window and its current
reconstruction top-left, a 256 -> k -> 256 bottleneck sketch top-right whose
middle bar is drawn to scale, and one chart below that changes with the
step (`bottom`): the real PCA error curve, the trivial-model MSE per target
(Meeting01's B2), the measured per-epoch cost, or the verdict panel.

`fast_clear` keeps scales, ticks and labels between frames, and the bottom
inset switches between a log-x curve and bar charts -- so every bottom mode
sets its own scale, ticks, labels and title on every frame.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon, Rectangle

from efficient_nn_lab.app.theme import ACCENT_COLOR, NEUTRAL_COLOR, SNN_COLOR, TEXT_COLOR

_RECON_COLOR = ACCENT_COLOR
_BOTTOM_RECT = (0.09, 0.09, 0.86, 0.31)
_DOMAINS = ((16, "áudio: 16"), (64, "EEG: 64"))


class AutoencoderComparisonRendererMixin:
    _AEC_AX_RECT = (0.0, 0.0, 1.0, 1.0)

    def _render_autoencoder_comparison_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        self._ax.text(
            0.5, 0.965, "Autoencoders: mesma janela, mesmo gargalo, mesmo alvo", ha="center", va="center",
            fontsize=10, weight="bold",
        )
        self._aec_signal(values)
        self._aec_bottleneck(values)
        bottom = self._get_inset("aec_error", rect=_BOTTOM_RECT)
        {
            "curve": self._aec_curve,
            "targets": self._aec_targets,
            "cost": self._aec_cost,
            "verdict": self._aec_verdict,
        }[str(values["bottom"])](bottom, values)

    # -- top left: the window and its reconstruction ----------------------
    def _aec_signal(self, values: dict[str, object]) -> None:
        top = self._get_inset("aec_signal", rect=(0.07, 0.56, 0.6, 0.33))
        window = np.asarray(values["window"], dtype=float)
        t = np.arange(len(window))
        top.plot(t, window, color=TEXT_COLOR, linewidth=1.2, zorder=3)
        reveal = float(values["recon_reveal"])
        title = "janela de teste (preto): 256 amostras, média 0, desvio 1"
        if reveal > 0.02:
            recon = np.asarray(values["recon"], dtype=float)
            top.plot(t, recon, color=_RECON_COLOR, linewidth=1.7, alpha=reveal, zorder=4)
            error_reveal = float(values["error_reveal"])
            if error_reveal > 0.02:
                top.fill_between(t, window, recon, color=_RECON_COLOR, alpha=0.25 * error_reveal, linewidth=0, zorder=2)
            sample_reveal = float(values["sample_reveal"])
            if sample_reveal > 0.02:
                i = int(round(float(values["sample"])))
                x, guess = window[i], recon[i]
                top.plot([i, i], [x, guess], color=SNN_COLOR, linewidth=2.2, alpha=sample_reveal, zorder=5)
                top.annotate(
                    f"amostra {i}: erro² = {(x - guess) ** 2:.2f}", (i, (x + guess) / 2), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=SNN_COLOR, weight="bold",
                    alpha=sample_reveal, zorder=6,
                )
            title = f"original (preto) × {values['recon_label']} (âmbar): MSE = {np.mean((window - recon) ** 2):.3f}"
        top.set_title(title, fontsize=8.5)
        span = float(np.abs(window).max()) * 1.15
        top.set_ylim(-span, span)
        top.set_xlim(0, len(window) - 1)
        top.set_xlabel("amostra", fontsize=7, labelpad=1)
        top.tick_params(labelsize=6.5)

    # -- top right: 256 -> k -> 256, the latent bar drawn to scale ----------
    def _aec_bottleneck(self, values: dict[str, object]) -> None:
        ax = self._ax
        n = len(np.asarray(values["window"]))
        sweeping = float(values["curve_upto"]) >= 1.0
        k = int(round(float(values["k_now"] if sweeping else values["k_target"])))
        x_in, x_lat, x_out, w = 0.735, 0.845, 0.955, 0.028
        y_mid, full = 0.735, 0.27
        h_lat = max(0.008, full * k / n)
        for x0, x1, h0, h1 in ((x_in, x_lat, full, h_lat), (x_lat, x_out, h_lat, full)):
            ax.add_patch(Polygon(
                [(x0 + w / 2, y_mid - h0 / 2), (x0 + w / 2, y_mid + h0 / 2),
                 (x1 - w / 2, y_mid + h1 / 2), (x1 - w / 2, y_mid - h1 / 2)],
                closed=True, facecolor=NEUTRAL_COLOR, alpha=0.14, edgecolor="none",
            ))
        for x, h, color, top_label, bottom_label in (
            (x_in, full, TEXT_COLOR, str(n), "janela"),
            (x_lat, h_lat, _RECON_COLOR, str(k), "latente"),
            (x_out, full, _RECON_COLOR, str(n), "reconstrução"),
        ):
            ax.add_patch(Rectangle((x - w / 2, y_mid - h / 2), w, h, facecolor=color, alpha=0.85, edgecolor="none"))
            ax.text(x, y_mid + max(h, 0.02) / 2 + 0.012, top_label, ha="center", va="bottom", fontsize=8, weight="bold")
            ax.text(x, y_mid - full / 2 - 0.015, bottom_label, ha="center", va="top", fontsize=7.5)
        ax.text((x_in + x_lat) / 2, y_mid + full / 2 + 0.02, "codifica", ha="center", fontsize=7, color=NEUTRAL_COLOR)
        ax.text((x_lat + x_out) / 2, y_mid + full / 2 + 0.02, "decodifica", ha="center", fontsize=7, color=NEUTRAL_COLOR)
        ratio = n / k
        ratio_text = f"{ratio:.0f}:1" if n % k == 0 else f"{ratio:.1f}:1"
        ax.text(x_lat, y_mid - full / 2 - 0.065, f"compressão {n}:{k} = {ratio_text}", ha="center", va="top", fontsize=7.5)

    # -- bottom modes --------------------------------------------------------
    def _aec_curve(self, bot, values: dict[str, object]) -> None:
        reveal = float(values["bottom_reveal"])
        k_all = np.asarray(values["k_all"])
        mse_all = np.asarray(values["mse_all"])
        mean_mse = float(values["mean_mse"])
        if bot.get_xscale() != "log":
            bot.set_xscale("log", base=2)
        if reveal > 0.02:
            bot.axhline(mean_mse, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.2, alpha=reveal)
            # right end: the PCA curve starts near this line at k = 1 and is
            # far below it by k = 128.
            bot.text(
                128, mean_mse + 0.03, f"palpite = média do treino (não aprendeu nada): MSE {mean_mse:.2f}",
                ha="right", va="bottom", fontsize=7, color=NEUTRAL_COLOR, alpha=reveal,
            )
        upto = float(values["curve_upto"])
        if upto >= 1.0:
            shown = k_all <= upto + 1e-9
            bot.plot(k_all[shown], mse_all[shown], color=_RECON_COLOR, linewidth=2.0)
            k = int(round(float(values["k_now"])))
            bot.plot([k], [mse_all[k - 1]], "o", color=_RECON_COLOR, markersize=7)
            bot.annotate(
                f"PCA, k = {k}: MSE {mse_all[k - 1]:.3f}", (k, mse_all[k - 1]), xytext=(6, 9),
                textcoords="offset points", fontsize=7.5, color=_RECON_COLOR, weight="bold",
            )
        domain_reveal = float(values["domain_reveal"])
        if domain_reveal > 0.02:
            for k, name in _DOMAINS:
                bot.axvline(k, color=TEXT_COLOR, linestyle=":", linewidth=1.1, alpha=domain_reveal)
                bot.text(k * 1.06, 0.55, name, ha="left", fontsize=7.5, weight="bold", alpha=domain_reveal)
        ticks = [1, 2, 4, 8, 16, 32, 64, 128]
        bot.set_xticks(ticks)
        bot.set_xticklabels([str(t) for t in ticks])
        bot.set_xlim(0.9, 140)
        bot.set_ylim(0, 1.2)
        bot.set_xlabel("tamanho do latente k (quantos números a janela vira)", fontsize=7, labelpad=1)
        bot.set_ylabel("MSE", fontsize=7)
        bot.set_title("Erro × tamanho do gargalo (PCA de verdade, na janela de teste)", fontsize=8.5)
        bot.tick_params(labelsize=6.5)

    def _aec_bars(self, bot, names, heights, colors, labels, ylim, ylabel: str, title: str) -> None:
        if bot.get_xscale() != "linear":
            bot.set_xscale("linear")
        for i, (h, color, label) in enumerate(zip(heights, colors, labels)):
            bot.add_patch(Rectangle((i - 0.3, 0), 0.6, h, facecolor=color, edgecolor="none", alpha=0.9))
            bot.text(i, h + ylim * 0.03, label, ha="center", va="bottom", fontsize=7.5, weight="bold")
        bot.set_xticks(range(len(names)))
        bot.set_xticklabels(names)
        bot.set_xlim(-0.6, len(names) - 0.4)
        bot.set_ylim(0, ylim)
        bot.set_xlabel("")
        bot.set_ylabel(ylabel, fontsize=7)
        bot.set_title(title, fontsize=8.5)
        bot.tick_params(labelsize=7)

    def _aec_targets(self, bot, values: dict[str, object]) -> None:
        trivial = dict(values["trivial"])
        names = [f"codificação {name}" for name in trivial]
        self._aec_bars(
            bot, names, list(trivial.values()), [NEUTRAL_COLOR, SNN_COLOR, SNN_COLOR],
            [f"{v:.4f}" if v < 0.01 else f"{v:.3f}" for v in trivial.values()], 1.25,
            "MSE sem aprender nada",
            "Modelo inútil, alvo = a própria entrada codificada: o MSE só mede a variância do alvo (bug B2)",
        )

    def _aec_cost(self, bot, values: dict[str, object]) -> None:
        seconds = dict(values["epoch_seconds"])
        names = list(seconds)
        colors = [SNN_COLOR if name.startswith("SNN") else NEUTRAL_COLOR for name in names]
        self._aec_bars(
            bot, names, list(seconds.values()), colors, [f"{v:g} s" for v in seconds.values()],
            max(seconds.values()) * 1.3, "segundos por época",
            "Custo medido por época no software/nn (CPU, lote 1, redes típicas, 2026-09-23)",
        )

    def _aec_verdict(self, bot, values: dict[str, object]) -> None:
        names = ["SNN-AE", "LSTM-AE", "GRU-AE", "Transformer-AE"]
        self._aec_bars(
            bot, names, [0.0] * len(names), [NEUTRAL_COLOR] * len(names), [""] * len(names), 1.25,
            "MSE no grupo de teste",
            "Quem vence? Só o experimento LOSO preenche estas barras",
        )
        mean_mse = float(values["mean_mse"])
        k = int(round(float(values["k_target"])))
        pca_mse = float(np.asarray(values["mse_all"])[k - 1])
        for i in range(len(names)):
            bot.text(i, 0.6, "?", ha="center", va="center", fontsize=18, color=NEUTRAL_COLOR, weight="bold")
        bot.axhline(mean_mse, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.2)
        bot.text(len(names) - 0.45, mean_mse + 0.02, f"média: {mean_mse:.2f}", ha="right", va="bottom", fontsize=7, color=NEUTRAL_COLOR)
        bot.axhline(pca_mse, color=_RECON_COLOR, linewidth=1.8)
        bot.text(
            len(names) - 0.45, pca_mse + 0.02, f"PCA, k = {k}: {pca_mse:.3f} — para valer, ficar abaixo",
            ha="right", va="bottom", fontsize=7, color=_RECON_COLOR, weight="bold",
        )
