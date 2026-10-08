"""Demonstração — a regra da cadeia como uma máquina de Rube Goldberg.

`chain_rule_layers.py` já faz a derivação rigorosa, bloco por bloco, dos
cinco fatores de ∂L/∂w1. O que falta ANTES dela é a intuição: por que
multiplicar cinco derivadas locais em fila é razoável, antes de ver o nome
de cada uma. Esta demo responde isso com uma imagem física -- uma bolinha
que nasce do erro da rede e atravessa cinco engenhocas (funil, rampa,
alavanca, rampa, roldana), uma por fator, saindo no balde final com
∂L/∂w1. Nenhuma matemática nova: `compute_chain_1_1_1` é a MESMA função que
`chain_rule_layers.py` usa, na mesma rede 1->1->1 -- as duas demos têm que
concordar por construção, não por coincidência (ver
`tests/test_backprop.py`'s `test_rube_goldberg_matches_chain_layers_demo`).

Por que uma máquina, e não só mais um diagrama de blocos: o aluno já viu o
diagrama de blocos NA OUTRA demo. O ganho de reapresentar os mesmos cinco
números como um sistema mecânico é que "multiplicar em fila" vira algo que
se vê acontecer -- a bolinha físicamente encolhe nas rampas (fator < 1) e
pode trocar de lado na alavanca (fator negativo), em vez de ser só um
produto de símbolos. Colocada ANTES de `chain_rule_layers.py` de propósito
(ESPECIFICACAO_DLVL.md #42 -- intuição antes do rigor).

Interação por clique: ao contrário de toda outra demo (que só usa os
controles da barra lateral), esta aceita clique na própria tela para
empurrar a bolinha para a próxima engenhoca -- literalmente "dar um
empurrão na máquina" em vez de apertar um botão genérico. Implementado em
widgets/neuron_view.py (`advance_requested`, emitido só quando o kind
renderizado é "rube_goldberg"); os controles de sempre (Próximo/Play/Passo)
continuam funcionando em paralelo, sem exigir o clique.

Números fixos e determinísticos (ESPECIFICACAO_DLVL.md #35).
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.backprop.demos.chain_rule_layers import compute_chain_1_1_1
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence, slider

#: Quantos elos (= engenhocas) a cadeia tem nesta rede 1->1->1 -- usado para
#: dimensionar os arrays one-hot `station_fired`/`station_glow`. O renderer
#: (widgets/renderers/rube_goldberg.py) não importa esta constante: ele
#: deriva o mesmo número de `len(chain_values)`, mantendo física (posições)
#: e semântica (quantos elos) como dois donos independentes que só
#: concordam porque os dois contam o mesmo tuple.
N_STATIONS = 5

#: Os cinco fatores de ∂L/∂w1, na ordem em que a cadeia os concatena (da
#: perda para o peso) -- mesma ordem e mesmos nomes de chain_rule_layers.py,
#: para as duas demos lerem como "a mesma cadeia, duas peles".
CHAIN_NAMES = ("∂L/∂a2", "σ'(z2)", "w2", "σ'(z1)", "x")

class RubeGoldbergChainDemo(DemoModule):
    title = "Backprop -> A regra da cadeia como máquina de Rube Goldberg"
    slug = "backprop.rube_goldberg"
    description = (
        "A mesma rede 1->1->1 e os mesmos cinco fatores de ∂L/∂w1 da demo "
        "'Camadas e a regra da cadeia', agora como uma máquina: uma bolinha nasce do "
        "erro da rede e atravessa cinco engenhocas -- funil, rampa, alavanca, rampa, "
        "roldana -- uma por fator, encolhendo nas rampas (fator < 1) e podendo trocar "
        "de lado na alavanca (peso negativo), até cair no balde final com ∂L/∂w1. "
        "Clique na tela para empurrar a bolinha para a próxima engenhoca."
    )

    def __init__(self) -> None:
        self.target = 0.2
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "target": slider("Alvo (0-1)", 0.05, 0.95, 0.05, self.target),
        }

    def _build_frames(self) -> list[Frame]:
        c = compute_chain_1_1_1(self.target)
        chain_values = (c["dL_da2"], c["sp2"], c["dz2_da1"], c["sp1"], c["dz1_dw1"])
        chain_partials = tuple(float(np.prod(chain_values[: i + 1])) for i in range(N_STATIONS))

        def onehot(index: int) -> np.ndarray:
            spot = np.zeros(N_STATIONS)
            spot[index] = 1.0
            return spot

        no_glow = np.zeros(N_STATIONS)
        all_glow = np.ones(N_STATIONS)
        no_fired = np.zeros(N_STATIONS)

        frames: list[Frame] = []

        def snap(label: str, explanation: str, equation: str, **overrides: object) -> None:
            values: dict[str, object] = {
                "kind": "rube_goldberg",
                "target": self.target, "loss": c["loss"],
                "chain_names": CHAIN_NAMES, "chain_values": chain_values,
                "chain_partials": chain_partials, "g_w1": chain_partials[-1],
                "ball_progress": 0.0, "ball_value": 0.0, "ball_reveal": 0.0,
                "bucket_reveal": 0.0,
                "station_fired": no_fired, "station_glow": no_glow,
            }
            values.update(overrides)
            frames.append(Frame(label, values, explanation, equation))

        snap(
            "A máquina",
            "Esta é a mesma regra da cadeia da demo 'Camadas e a regra da cadeia', só que em "
            "forma de máquina: uma bolinha nasce do erro da rede e atravessa cinco engenhocas, "
            "uma por fator da cadeia. Cada engenhoca multiplica a bolinha pelo SEU fator -- "
            "exatamente como cada bloco daquela outra demo. Clique em qualquer lugar da tela "
            "(ou em 'Próximo') para dar o primeiro empurrão.",
            "∂L/∂w1 = ∂L/∂a2 · σ'(z_2) · w_2 · σ'(z_1) · x",
        )
        snap(
            "A bolinha cai no funil",
            f"O funil recebe o erro da rede: ∂L/∂a2 = {chain_values[0]:+.4f}. É o primeiro fator "
            "da cadeia -- a bolinha nasce com este valor.",
            "∂L/∂a2 = a_2 - alvo",
            ball_progress=0.0, ball_value=chain_partials[0], ball_reveal=1.0,
            station_fired=onehot(0), station_glow=onehot(0),
        )
        snap(
            "Desce a rampa da camada 2",
            f"A rampa multiplica pela derivada da ativação, σ'(z2) = {chain_values[1]:.4f} -- "
            "sempre um número entre 0 e 0,25, então a bolinha SEMPRE encolhe aqui. Produto até "
            f"agora: {chain_partials[0]:+.4f} · {chain_values[1]:.4f} = {chain_partials[1]:+.5f}.",
            "σ'(z_2) = a_2 (1 - a_2)",
            ball_progress=1.0, ball_value=chain_partials[1], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1)), station_glow=onehot(1),
        )
        sign_note = (
            "Como w2 é negativo, a alavanca JOGA a bolinha para o outro lado -- o sinal do "
            "gradiente inverte."
            if chain_values[2] < 0
            else "w2 é positivo aqui, então a alavanca não troca o lado da bolinha."
        )
        snap(
            "A alavanca do peso w2",
            f"A alavanca multiplica pelo peso w2 = {chain_values[2]:+.2f}. {sign_note} Produto "
            f"até agora: {chain_partials[2]:+.5f}.",
            "∂z2/∂a1 = w_2",
            ball_progress=2.0, ball_value=chain_partials[2], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2)), station_glow=onehot(2),
        )
        snap(
            "Desce a rampa da camada 1",
            f"Segunda rampa, derivada local da SUA PRÓPRIA ativação: σ'(z1) = {chain_values[3]:.4f}. "
            "Não é a mesma rampa da camada 2 -- cada ativação tem a sua. Produto até agora: "
            f"{chain_partials[3]:+.6f}.",
            "σ'(z_1) = a_1 (1 - a_1)",
            ball_progress=3.0, ball_value=chain_partials[3], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3)), station_glow=onehot(3),
        )
        snap(
            "A roldana da entrada",
            "Última engenhoca: multiplica pela entrada que passou por w1, x = "
            f"{chain_values[4]:+.2f}. É o quinto e último fator -- produto final: "
            f"{chain_partials[4]:+.6f}.",
            "∂z1/∂w1 = x",
            ball_progress=4.0, ball_value=chain_partials[4], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3) + onehot(4)),
            station_glow=onehot(4),
        )
        snap(
            "O balde final: ∂L/∂w1",
            f"A bolinha cai no balde com o produto dos cinco fatores: ∂L/∂w1 = "
            f"{chain_partials[4]:+.6f}. É exatamente o mesmo número que a demo 'Camadas e a "
            "regra da cadeia' calcula bloco por bloco -- duas máquinas, um resultado.",
            "∂L/∂w1 = δ_1 · x",
            ball_progress=5.0, ball_value=chain_partials[4], ball_reveal=1.0, bucket_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3) + onehot(4)),
            station_glow=no_glow,
        )
        snap(
            "Cada engenhoca, uma derivada local",
            "Olhando a esteira inteira: cada engenhoca é a derivada local de UM bloco da rede, "
            "na MESMA ordem em que a regra da cadeia as multiplica. Para ver de onde cada "
            "engenhoca vem -- qual bloco, qual neurônio -- a próxima demo faz a mesma viagem "
            "devagar, passo a passo.",
            "∂L/∂w1 = ∂L/∂a2 · σ'(z_2) · w_2 · σ'(z_1) · x",
            ball_progress=5.0, ball_value=chain_partials[4], ball_reveal=1.0, bucket_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3) + onehot(4)),
            station_glow=all_glow,
        )

        return build_sequence(frames, steps=8)
