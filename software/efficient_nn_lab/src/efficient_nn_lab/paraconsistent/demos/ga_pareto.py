"""Demonstração — busca multiobjetivo de arquiteturas (NSGA-II) usando
D_penalized como um dos objetivos.

NÃO está na monografia da tese: a tese usa D_penalized para ranquear
CONFIGURAÇÕES DE EXTRAÇÃO DE FEATURES (ver paraconsistent.plane /
paraconsistent.dpenalized). Quem usa o mesmo número como objetivo de uma
busca genética de ARQUITETURAS é o experimento `paraconsistentGA` do
software/nn (objetivos `{d_penalized_mean, inference_cost}` em
`lib/include/GaFitness.hpp`, dominância restrita de Deb em
`include/ga/Nsga2Core.hpp`). Os pontos são sintéticos -- ver
`paraconsistent/ga_synthetic.py`.

The steps build the vocabulary one piece at a time on generation 0
(dominance with three named points, then the front, then the latency
ceiling sliding in under Deb's rule) before any generation runs; the front
and the feasibility marks are recomputed by the renderer on every frame, so
they genuinely change as points move and as the ceiling passes them.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, slider, transition
from efficient_nn_lab.paraconsistent.ga_synthetic import (
    Population,
    constraint_violation,
    dominates,
    first_front_mask,
    synthetic_generations,
)

_N_GENERATIONS = 5
_POP_SIZE = 12
_STEPS = 12  # tween frames between two concept steps
_GEN_STEPS = 8  # tween frames per generation
#: Right edge of the cost axis: the ceiling line slides in from here, past
#: every synthetic point (the costs stop at 1.25), so nothing is infeasible
#: before the ceiling step.
X_MAX = 1.3
# A mid-size encoder of the real run, 256 -> 128 -> 64 -> 32 with T = 16:
# (256*128 + 128*64 + 64*32) MACs per frame x 16 frames
# (GaGenome.cpp encoder_macs_per_frame / inference_cost_proxy, input = 256).
_EXAMPLE_MACS = (256 * 128 + 128 * 64 + 64 * 32) * 16

_DOMINANCE_EQUATION = "A \\text{ domina } B:\\ c_A \\leq c_B \\text{ e } D_A \\leq D_B,\\ \\text{uma delas estrita}"


def dominance_example(pop: Population) -> tuple[int, int, int]:
    """(A, B, C) for the dominance step: A is the front point that dominates
    the most others, B the point A beats by the widest margin in BOTH
    objectives, and C a front neighbor of A -- a trade-off, not a
    domination (preferring the costlier neighbor, so C is the better-D one)."""
    cost, d = pop.cost, pop.d_penalized
    n = len(cost)
    front = [int(i) for i in np.flatnonzero(first_front_mask(cost, d))]

    def beaten(i: int) -> list[int]:
        return [j for j in range(n) if dominates(cost[i], d[i], cost[j], d[j])]

    a = max(front, key=lambda i: (len(beaten(i)), -cost[i]))
    b = max(beaten(a), key=lambda j: min(cost[j] - cost[a], d[j] - d[a]))
    by_cost = sorted(front, key=lambda i: cost[i])
    k = by_cost.index(a)
    c = by_cost[k + 1] if k + 1 < len(by_cost) else by_cost[k - 1]
    return int(a), int(b), int(c)


def _feasible_front(pop: Population, ceiling: float) -> np.ndarray:
    return first_front_mask(pop.cost, pop.d_penalized, constraint_violation(pop.cost, ceiling))


def _pt(pop: Population, i: int) -> str:
    return f"custo {pop.cost[i]:.2f}, D {pop.d_penalized[i]:.2f}"


class ParaconsistentGaParetoDemo(DemoModule):
    title = "Paraconsistente -> Busca genética de arquiteturas (extensão)"
    slug = "paraconsistent.ga_pareto"
    description = (
        "Fora da monografia: o experimento paraconsistentGA do software/nn usa o D_penalized da "
        "tese como um dos dois objetivos de uma busca NSGA-II por arquiteturas de autoencoder; o "
        "outro é o custo de inferência. Dominância, fronteira de Pareto, teto de latência e "
        "gerações -- com dados sintéticos ilustrativos, não uma busca real."
    )

    def __init__(self) -> None:
        self.latency_ceiling = 0.7
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "latency_ceiling": slider("teto de latência (custo)", 0.3, 1.2, 0.05, self.latency_ceiling),
        }

    def _build_frames(self) -> list[Frame]:
        ceiling = float(self.latency_ceiling)
        gens = synthetic_generations(n_generations=_N_GENERATIONS, pop_size=_POP_SIZE, latency_ceiling=ceiling)
        first, last = gens[0], gens[-1]
        a, b, c = dominance_example(first)
        plain_front = first_front_mask(first.cost, first.d_penalized)
        last_front = _feasible_front(last, ceiling)
        infeasible0 = constraint_violation(first.cost, ceiling) > 0
        menu = np.flatnonzero(last_front)
        lo = int(menu[np.argmin(last.cost[menu])])
        hi = int(menu[np.argmin(last.d_penalized[menu])])

        def best_feasible_d(pop: Population) -> float:
            ok = constraint_violation(pop.cost, ceiling) <= 0
            return float(pop.d_penalized[ok].min())

        def frame(label: str, explanation: str, equation: str, pop: Population, gen: int,
                  checkpoint: bool = True, **reveals: float) -> Frame:
            values: dict[str, object] = {
                "kind": "paraconsistent_ga_pareto",
                "cost": pop.cost, "d": pop.d_penalized,
                "generation": float(gen), "n_generations": _N_GENERATIONS,
                "x_max": X_MAX, "ceiling_x": X_MAX, "ceiling_value": ceiling, "ceiling_reveal": 0.0,
                "front_reveal": 0.0, "marks_reveal": 0.0, "box_reveal": 0.0, "menu_reveal": 0.0,
                "focus": (a, b, c), "menu": (lo, hi),
            }
            values.update(reveals)
            return Frame(label, values, explanation, equation, is_checkpoint=checkpoint)

        intro = frame(
            "Dois objetivos que brigam",
            "Cada ponto é uma arquitetura de autoencoder (larguras das camadas, time_steps, "
            "codificação) já treinada e pontuada. Queremos duas coisas: features boas (D_penalized "
            "baixo, eixo y) e inferência barata (eixo x: MACs do codificador × time_steps, "
            "normalizado). Em geral brigam — redes maiores tendem a dar features melhores, mas custam "
            "mais. É o paraconsistentGA do software/nn, fora da monografia; estes "
            f"{_POP_SIZE} pontos são sintéticos.",
            "\\min\\ (D_penalized,\\ custo)",
            first, 0,
        )
        c_relation = (
            "C é melhor, A é mais barata" if first.cost[c] > first.cost[a] else "C é mais barata, A é melhor"
        )
        dominance = frame(
            "Dominância: não perder em nada, ganhar em algo",
            f"A domina B quando não perde em nenhum objetivo e ganha em pelo menos um. A "
            f"({_pt(first, a)}) × B ({_pt(first, b)}): A é mais barata E melhor — escolher B seria "
            f"desperdício; o retângulo sombreado é tudo que A domina. A × C ({_pt(first, c)}): "
            f"{c_relation} — nenhuma domina a outra. Isso não é empate: é uma troca.",
            _DOMINANCE_EQUATION,
            first, 0, marks_reveal=1.0, box_reveal=1.0,
        )
        front = frame(
            "Fronteira de Pareto: quem ninguém domina",
            f"Riscando todo ponto dominado sobra a fronteira de Pareto: {int(plain_front.sum())} dos "
            f"{_POP_SIZE} pontos — A e C estão nela, B não. Cada ponto da fronteira é a melhor escolha "
            "para algum orçamento, e andar sobre ela é sempre trocar custo por qualidade. Um "
            "otimizador de um objetivo só devolveria um ponto; o NSGA-II procura a fronteira inteira.",
            "\\text{fronteira} = \\{x : \\text{nenhum } y \\text{ domina } x\\}",
            first, 0, marks_reveal=1.0, front_reveal=1.0,
        )

        lost = [int(i) for i in np.flatnonzero(plain_front & infeasible0)]
        best_overall = int(np.argmin(first.d_penalized))
        names = {a: "A", c: "C"}
        if not lost:
            lost_clause = ", mas nenhum era da fronteira: ela não muda"
        else:
            lost_clause = (
                "; um deles era da fronteira e sai" if len(lost) == 1 else f"; {len(lost)} eram da fronteira e saem"
            )
            named = all(i in names for i in lost)
            if named:
                lost_clause += ": " + " e ".join(names[i] for i in lost)
            if best_overall in lost:
                best_d = f"melhor D de todos ({first.d_penalized[best_overall]:.2f})"
                if named and len(lost) == 1:
                    lost_clause += f", o de {best_d}"
                elif named:
                    lost_clause += f" — {names[best_overall]} tinha o {best_d}"
                else:
                    lost_clause += f", inclusive o de {best_d}"
        n_inf = int(infeasible0.sum())
        ceiling_step = frame(
            "Teto de latência: viável sempre vence",
            f"O dispositivo impõe um teto: custo acima de {ceiling:.2f} não cabe no orçamento de "
            "latência. Regra de Deb: viável domina inviável, por melhor que seja o D do inviável; "
            "entre inviáveis vence quem estoura menos; entre viáveis, Pareto. "
            f"{n_inf} {'ponto passa' if n_inf == 1 else 'pontos passam'} do teto{lost_clause}. No "
            "software/nn quem estoura o teto nem é treinado: sai na triagem pelo custo estrutural.",
            "violacao = \\max(0,\\ custo - teto)",
            first, 0, ceiling_x=ceiling, ceiling_reveal=1.0, front_reveal=1.0,
        )

        generations_text = (
            "Cada geração do NSGA-II escolhe pais por torneio (vence a fronteira melhor; no empate, o "
            "ponto mais isolado), gera filhos por cruzamento e mutação (alargar, estreitar, inserir ou "
            "remover camada), junta pais e filhos e mantém os melhores — elitismo. Em "
            f"{_N_GENERATIONS - 1} gerações o melhor D viável cai de {best_feasible_d(first):.2f} "
            f"para {best_feasible_d(last):.2f}. Simplificação: aqui cada ponto desliza até um "
            "descendente; no algoritmo real os piores somem."
        )
        generations_equation = "P_{t+1} = \\text{melhores}(P_t \\cup \\text{filhos}_t)"
        gen_frames = [
            frame(
                f"Geração {g + 1}/{_N_GENERATIONS}: a população desce até a fronteira",
                generations_text, generations_equation, gens[g], g,
                checkpoint=g == _N_GENERATIONS - 1,
                ceiling_x=ceiling, ceiling_reveal=1.0, front_reveal=1.0,
            )
            for g in range(1, _N_GENERATIONS)
        ]
        last_gen = gen_frames[-1]
        menu_step = frame(
            "Um cardápio, não um vencedor",
            f"O resultado é a fronteira inteira: {int(last_front.sum())} opções. Na ponta barata, "
            f"{_pt(last, lo)}; na ponta boa, {_pt(last, hi)}. Quem escolhe é o orçamento do "
            "dispositivo. Por que não somar os objetivos com pesos? As escalas não conversam: D vai "
            "de 0 a 2, e o custo real de um codificador 256 → 128 → 64 → 32 com T = 16 é "
            f"{_EXAMPLE_MACS} MACs. A soma só enxergaria o custo — e nada na tela avisaria.",
            f"0.5 · D + 0.5 · MACs:\\ D \\leq 2,\\ MACs = {_EXAMPLE_MACS}",
            last, _N_GENERATIONS - 1, ceiling_x=ceiling, ceiling_reveal=1.0, front_reveal=1.0, menu_reveal=1.0,
        )

        frames = [intro]
        for start, end in ((intro, dominance), (dominance, front), (front, ceiling_step)):
            frames += transition(start, end, _STEPS)
            frames.append(end)
        previous = ceiling_step
        for gen_frame in gen_frames:
            frames += transition(previous, gen_frame, _GEN_STEPS)
            frames.append(gen_frame)
            previous = gen_frame
        frames += transition(last_gen, menu_step, _STEPS)
        frames.append(menu_step)
        return frames
