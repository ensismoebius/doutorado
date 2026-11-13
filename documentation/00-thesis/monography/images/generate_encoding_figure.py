"""
Gera ``spikeEncodings.pdf``: as tres codificacoes temporais do SNN-AE aplicadas ao mesmo
vetor de entrada (secao do SNN-AE no capitulo 8).

As regras sao as de ``spike_frame()`` em
``software/nn/src/experiments/thesis/lib/src/ThesisFeatureExtractionInternal.hpp``:

    poisson  quadro t: 1 se U(0,1) < v, sorteio novo a cada passo        (T quadros)
    latency  quadro t: 1 se t >= round((1 - v)(T - 1)), isto e, a partir do instante
             de disparo o componente continua disparando                   (T quadros)
    direct   um unico quadro analogico com o proprio v                     (1 quadro)

``round`` e ``std::lround`` (metade para longe do zero), reproduzido com floor(x + 0.5):
o ``round`` do Python arredonda metade para o par. O sorteio do poisson usa o numpy, nao o
mt19937 do C++, entao a figura mostra *um* sorteio, nao o da execucao; latency e direct
sao deterministicos e iguais aos do codigo. A coluna a direita de cada painel e a media
sobre os quadros, o que o LIF integra.

Uso:
    python3 generate_encoding_figure.py [diretorio_de_saida]
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

T = 16                      # time_steps dos perfis (00_thesis_generate_profiles.py)
VALUES = np.linspace(0.0, 1.0, 8)[::-1]   # de cima para baixo: do maior ao menor
SEED = 3


def poisson(v, rng):
    return (rng.random((len(v), T)) < v[:, None]).astype(float)


def latency(v):
    last = T - 1
    t_spike = np.array([math.floor((1.0 - x) * last + 0.5) for x in v])
    return (np.arange(T)[None, :] >= t_spike[:, None]).astype(float), t_spike


def block(ax, x0, frames, title, last_label):
    """frames (rows, cols) drawn as cells of side 1, top row first, from x = x0; the mean
    over the frames is printed to the right."""
    rows, cols = frames.shape
    for r in range(rows):
        for c in range(cols):
            g = 1.0 - frames[r, c]
            ax.add_patch(Rectangle((x0 + c, r), 1, 1, facecolor=(g, g, g),
                                   edgecolor="0.7", linewidth=0.4))
    ax.add_patch(Rectangle((x0, 0), cols, rows, fill=False, edgecolor="black", linewidth=0.8))
    ax.text(x0 + cols / 2, -1.4, title, ha="center", va="bottom", fontsize=8.5)
    ax.text(x0 + cols + 0.4, -0.3, "média", ha="left", va="bottom", fontsize=7)
    for r, m in enumerate(frames.mean(axis=1)):
        ax.text(x0 + cols + 0.4, r + 0.5, comma(m), ha="left", va="center", fontsize=7)
    ax.text(x0 + 0.5, rows + 0.3, "0", ha="center", va="top", fontsize=7)
    if cols > 1:
        ax.text(x0 + cols - 0.5, rows + 0.3, last_label, ha="center", va="top", fontsize=7)
    ax.text(x0 + cols / 2, rows + 1.3, "passo $t$", ha="center", va="top", fontsize=7.5)


def comma(x):
    return f"{x:.2f}".replace(".", ",")


def main(out_dir):
    v = VALUES
    p = poisson(v, np.random.default_rng(SEED))
    lat, t_spike = latency(v)
    direct = v[:, None]
    rows = len(v)

    gap = 5.5
    x_p, x_l = 0, T + gap
    x_d = 2 * (T + gap)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.text(-1.6, -0.3, "$v$", ha="center", va="bottom", fontsize=8)
    for r, x in enumerate(v):
        ax.text(-1.6, r + 0.5, comma(x), ha="center", va="center", fontsize=7)
    block(ax, x_p, p, "(a) poisson: taxa", str(T - 1))
    block(ax, x_l, lat, "(b) latency: instante", str(T - 1))
    for r, ts in enumerate(t_spike):
        ax.plot([x_l + ts + 0.08] * 2, [r + 0.1, r + 0.9], color="tab:red", linewidth=1.4)
    block(ax, x_d, direct, "(c) direct", "")
    ax.set_xlim(-3.2, x_d + 4.2)
    ax.set_ylim(rows + 2.2, -2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "spikeEncodings.pdf")
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
