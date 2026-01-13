import matplotlib.pyplot as plt
import numpy as np
import torch

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
    output_file: str = "result_pipeline_wpt_snn.png",
):
    print("[Visualização] Gerando gráficos...")
    
    # Prepara dados
    # Características: [num_janelas, num_bandas]
    matriz_caracteristicas = np.vstack(lista_caracteristicas)
    # Spikes: [num_janelas, num_neuronios_saida]
    matriz_spikes = torch.vstack(lista_spikes).detach().cpu().numpy()
    
    # Eixo X: tempo (segundos) se tivermos sample_rate/hop_size; senão, índice da janela.
    num_janelas = matriz_caracteristicas.shape[0]
    if sample_rate is not None and hop_size is not None and window_size is not None:
        total_time = ((num_janelas - 1) * hop_size + window_size) / float(sample_rate)
        x_extent = (0.0, float(total_time))
        x_label = "Tempo (s)"
        x_line = np.linspace(x_extent[0], x_extent[1], num_janelas)
    else:
        x_extent = (0.0, float(num_janelas))
        x_label = "Índice da janela"
        x_line = np.arange(num_janelas, dtype=float)

    # Eixo Y das características: frequência (Hz, aproximado) se tivermos sample_rate; senão índice da banda.
    num_bandas_calc = matriz_caracteristicas.shape[1]
    if sample_rate is not None:
        y_extent_feat = (0.0, float(sample_rate) / 2.0)
        y_label_feat = "Frequência (Hz, aprox.)"
    else:
        y_extent_feat = (0.0, float(num_bandas_calc))
        y_label_feat = "Banda (índice)"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Cabeçalho didático (parâmetros do experimento)
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

    fig.suptitle(
        "Pipeline WPT → SNN (visualização didática)\n" + " | ".join(parts), fontsize=12
    )
    
    # Plot 1: Características (energia WPT)
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
    
    # Plot 2: Spikes (mapa)
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
    fig.colorbar(im2, ax=ax2, label="Spike (0/1)")

    # Plot 3: Resumo (taxa de spikes por janela)
    spikes_por_janela = matriz_spikes.sum(axis=1)
    ax3.plot(x_line, spikes_por_janela, color="black", linewidth=1.5)
    ax3.set_ylabel("Spikes/Janela")
    ax3.set_xlabel(x_label)
    ax3.set_title("Resumo: quantidade de spikes por janela")
    ax3.grid(True, alpha=0.3)

    # Caixa de estatísticas rápidas (didática)
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
    
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_file, dpi=150)
    print(f"[Visualização] Salvo em: {output_file}")
