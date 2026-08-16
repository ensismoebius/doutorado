"""
Regenera as figuras de potencial de membrana do LIF usadas na tese
(``membranePotentialFull.pdf``, ``membranePotentialDecay.pdf`` e
``membranePotentialIncrease.pdf``) com fontes vetoriais embutidas
(``pdf.fonttype=42``) em vez das fontes Type 3 do matplotlib.

Contexto
--------
Os PDFs originais (gerados com Matplotlib v3.6.3, script perdido) embutiam as
etiquetas de texto como fontes Type 3, o que degrada a qualidade de impressao
e ferramentas de preflight. Como o script gerador original nao existe, os
arquivos versionados foram preservados com aparência identica via conversao
de contorno (Ghostscript ``-dNoOutputFonts=true``) e sao, portanto, livres de
fontes Type 3 e visivelmente iguais aos originais.

Este script e a implementacao de referencia: documenta a dinamica LIF usada
nas figuras e produz equivalentes fieis. Rode apenas se os arquivos originais
forem perdidos ou se uma nova versao for necessaria; a saída pode diferir
levemente na estilização (versão do matplotlib).

Dinamica LIF (listings da tese)
-------------------------------
Atualizacao por Euler com ``dt=1``::

    tau = R * C
    V_mem += (dt / tau) * (-V_mem + I_in * R)
    if V_mem > V_thresh:
        V_mem = 0            # reset_zero

Uso
---
    python generate_membrane_figures.py [diretorio_de_saida]
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

R = 5.0
C = 1.0
TAU = R * C
V_THRESH = 2.0
DT = 1.0


def lif_step(v_mem: float, i_in: float) -> float:
    v_mem += (DT / TAU) * (-v_mem + i_in * R)
    if v_mem > V_THRESH:
        v_mem = 0.0
    return v_mem


def simulate(times: range, i_in_at: Callable[[int], float]) -> list[float]:
    """Simula o LIF para cada instante em ``times``; ``i_in_at`` retorna a
    corrente (mA) no instante ``t``."""
    v = 0.0
    out: list[float] = []
    for t in times:
        v = lif_step(v, i_in_at(t))
        out.append(v)
    return out


def fig_full():
    """Duas janelas: potencial de membrana (topo) e corrente de entrada
    (base), com 0.5 mA injetados em t = 51..70."""
    times = range(0, 121)
    i_in_at = lambda t: 0.5 if 51 <= t <= 70 else 0.0
    v = simulate(times, i_in_at)
    i = [i_in_at(t) for t in times]

    fig, (axv, axi) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 4), constrained_layout=True
    )
    axv.plot(times, v)
    axv.set_title("LIF potential")
    axv.set_ylabel("Voltage (mV)")
    axv.set_ylim(0, 2)
    axv.set_yticks([0, 1, 2])
    axi.plot(times, i)
    axi.set_title("LIF input")
    axi.set_ylabel("Input current (mA)")
    axi.set_xlabel("Time")
    axi.set_ylim(0, 0.6)
    return fig


def fig_increase():
    """Potencial subindo em direção ao equilíbrio I_in*R = 5 V (I_in = 1 mA)."""
    times = range(0, 41)
    i_in_at = lambda t: 1.0
    v = simulate(times, i_in_at)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(times, v)
    ax.set_title("LIF potential increase")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    return fig


def fig_decay():
    """Decaimento exponencial do potencial com corrente nula a partir de
    V(0) ~ 0.77 V em direção a um pequeno vazamento residual."""
    times = range(0, 41)
    i_in_at = lambda t: 0.0
    v0 = 0.765
    v = []
    for t in times:
        v.append(v0 * (1.0 - 1.0 / TAU) ** t)
    fig, ax = plt.subplots(figsize=(8.71, 3.23))
    ax.plot(times, v)
    ax.set_title("LIF potential decay")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    return fig


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    figures = {
        "membranePotentialFull.pdf": fig_full,
        "membranePotentialIncrease.pdf": fig_increase,
        "membranePotentialDecay.pdf": fig_decay,
    }
    for name, builder in figures.items():
        builder().savefig(f"{out_dir}/{name}")
        plt.close("all")
        print(f"escreveu {out_dir}/{name}")


if __name__ == "__main__":
    main()
