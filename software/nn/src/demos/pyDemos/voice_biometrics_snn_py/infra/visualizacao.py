"""Rotinas de visualização didática da pipeline WPT -> SNN."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def _compute_x_axis(num_janelas, sample_rate, hop_size, window_size):
    # Eixo X: tempo (segundos) se tivermos sample_rate/hop_size; senão, índice da janela.
    if sample_rate is not None and hop_size is not None and window_size is not None:
        # Referencial do eixo X: início de cada janela.
        # Isso garante alinhamento perfeito entre mapas (imshow) e séries (plot), já que ambos
        # são indexados por janela.
        x_starts = (np.arange(num_janelas, dtype=float) * float(hop_size)) / float(
            sample_rate
        )
        # Se houver 1 janela apenas, cria uma extensão mínima para o imshow.
        x_extent = (
            float(x_starts[0]),
            float(x_starts[-1]) if num_janelas > 1 else float(x_starts[0]) + 1e-6,
        )
        x_label = "Tempo (s)"
        x_line = x_starts
    else:
        x_idx = np.arange(num_janelas, dtype=float)
        x_extent = (0.0, float(x_idx[-1]) if num_janelas > 1 else 1.0)
        x_label = "Índice da janela"
        x_line = x_idx
    return x_extent, x_label, x_line


def _compute_y_axis_features(num_bandas_calc, sample_rate):
    # Eixo Y das características: frequência (Hz, aprox.) se tivermos sample_rate; senão índice da banda.
    if sample_rate is not None:
        y_extent_feat = (0.0, float(sample_rate) / 2.0)
        y_label_feat = "Frequência (Hz, aprox.)"
    else:
        y_extent_feat = (0.0, float(num_bandas_calc))
        y_label_feat = "Banda (índice)"
    return y_extent_feat, y_label_feat


def _build_header_title(
    duration, sample_rate, window_size, hop_size, wavelet, wpt_level, num_bands, stateful
):
    # Cabeçalho didático (parâmetros do experimento).
    parts = []
    if duration is not None:
        parts.append(f"Duração={duration}s")
    if sample_rate is not None:
        parts.append(f"fs={sample_rate}Hz")
    if window_size is not None and hop_size is not None:
        overlap = 100.0 * (1.0 - (hop_size / float(window_size)))
        parts.append(f"Janela={window_size}, Hop={hop_size} (overlap≈{overlap:.0f}%)")
    if wavelet is not None:
        parts.append(f"Wavelet={wavelet}")
    if wpt_level is not None:
        parts.append(f"WPT_LEVEL={wpt_level}")
    if num_bands is not None:
        parts.append(f"Bandas={num_bands}")
    if stateful is not None:
        parts.append(f"SNN stateful={stateful}")

    return "Pipeline WPT → SNN (visualização didática)\n" + " | ".join(parts)


def _plot_features(fig, ax1, matriz_caracteristicas, x_extent, y_extent_feat, y_label_feat):
    # Plot 1: Características (energia WPT).
    # Para visualização, usamos log1p para comprimir a faixa dinâmica sem alterar a característica “real”.
    caracteristicas_vis = np.log1p(np.maximum(matriz_caracteristicas, 0.0))
    im1 = ax1.imshow(
        caracteristicas_vis.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="inferno",
        extent=[x_extent[0], x_extent[1], y_extent_feat[0], y_extent_feat[1]],
    )
    ax1.set_ylabel(y_label_feat)
    ax1.set_title("Energia WPT por banda (log(1+E) apenas para visualização)")
    fig.colorbar(im1, ax=ax1, label="log(1+Energia)")


def _plot_spikes_map(fig, ax2, matriz_spikes, x_extent):
    # Plot 2: Spikes (mapa).
    im2 = ax2.imshow(
        matriz_spikes.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=[x_extent[0], x_extent[1], 0.0, float(matriz_spikes.shape[1])],
    )
    ax2.set_ylabel("Neurônio (índice)")
    ax2.set_title("Atividade de spikes (saída da SNN)")
    fig.colorbar(im2, ax=ax2, label="Spikes (contagem)")


def _plot_spikes_summary(ax3, matriz_spikes, x_line, x_label):
    # Plot 3: Resumo (taxa de spikes por janela).
    spikes_por_janela = matriz_spikes.sum(axis=1)
    ax3.plot(
        x_line,
        spikes_por_janela,
        color="black",
        linewidth=1.2,
        marker=".",
        markersize=3,
    )
    ax3.set_ylabel("Spikes/Janela")
    ax3.set_xlabel(x_label)
    ax3.set_title("Resumo: quantidade de spikes por janela")
    ax3.grid(True, alpha=0.3)
    return spikes_por_janela


def _marcar_janelas_verticais(ax1, ax2, ax3, x_line, num_janelas, max_marcas_janelas):
    # Marcação das janelas (linhas verticais nos instantes de início).
    if max_marcas_janelas < 1:
        max_marcas_janelas = 1

    stride_mapas = int(np.ceil(num_janelas / float(max_marcas_janelas)))
    stride_mapas = max(1, stride_mapas)

    inicios = x_line

    # Nos mapas (ax1/ax2), subamostramos para não poluir.
    for idx in range(0, num_janelas, stride_mapas):
        x0 = float(inicios[idx])
        for ax in (ax1, ax2):
            ax.axvline(x0, color="white", linewidth=0.8, alpha=0.12)

    # No resumo (ax3), marcamos *todas* as janelas.
    y0, y1 = ax3.get_ylim()
    ax3.vlines(
        inicios,
        y0,
        y1,
        colors="0.2",
        linewidth=0.5,
        alpha=0.10,
    )
    ax3.set_ylim(y0, y1)


def _add_stats_box(ax3, matriz_caracteristicas, spikes_por_janela):
    # Caixa de estatísticas rápidas (didática).
    media_carac = float(np.mean(matriz_caracteristicas))
    desvio_carac = float(np.std(matriz_caracteristicas))
    spk_mean = float(np.mean(spikes_por_janela))
    spk_std = float(np.std(spikes_por_janela))
    stats = (
        f"Características: média={media_carac:.3g}, desvio={desvio_carac:.3g}\n"
        f"Spikes/janela: mean={spk_mean:.3g}, std={spk_std:.3g}"
    )
    ax3.text(
        0.01,
        0.98,
        stats,
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )


def plotar_resultados(
    lista_caracteristicas,
    lista_spikes,
    *,
    sample_rate: int | None = None,
    window_size: int | None = None,
    hop_size: int | None = None,
    duration: float | None = None,
    wavelet: str | None = None,
    wpt_level: int | None = None,
    num_bands: int | None = None,
    stateful: bool | None = None,
    marcar_janelas: bool = True,
    max_marcas_janelas: int = 40,
    output_file: str = "result_pipeline_wpt_snn.png",
):
    # Entrada esperada: listas de janelas (características e spikes agregados).
    print("[Visualização] Gerando gráficos...")

    # Prepara dados: empilha listas em matrizes 2D.
    # Características: [num_janelas, num_bandas]
    matriz_caracteristicas = np.vstack(lista_caracteristicas)
    # Spikes: [num_janelas, num_neuronios_saida]
    matriz_spikes = torch.vstack(lista_spikes).detach().cpu().numpy()

    num_janelas = matriz_caracteristicas.shape[0]
    x_extent, x_label, x_line = _compute_x_axis(
        num_janelas, sample_rate, hop_size, window_size
    )

    num_bandas_calc = matriz_caracteristicas.shape[1]
    y_extent_feat, y_label_feat = _compute_y_axis_features(num_bandas_calc, sample_rate)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    fig.suptitle(
        _build_header_title(
            duration, sample_rate, window_size, hop_size, wavelet, wpt_level, num_bands, stateful
        ),
        fontsize=12,
    )

    _plot_features(fig, ax1, matriz_caracteristicas, x_extent, y_extent_feat, y_label_feat)
    _plot_spikes_map(fig, ax2, matriz_spikes, x_extent)
    spikes_por_janela = _plot_spikes_summary(ax3, matriz_spikes, x_line, x_label)

    if marcar_janelas:
        _marcar_janelas_verticais(ax1, ax2, ax3, x_line, num_janelas, max_marcas_janelas)

    # Garante alinhamento explícito do eixo X entre todos os plots.
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(x_extent)

    _add_stats_box(ax3, matriz_caracteristicas, spikes_por_janela)

    # Ajusta layout e salva o arquivo de saída.
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_file, dpi=150)
    print(f"[Visualização] Salvo em: {output_file}")
