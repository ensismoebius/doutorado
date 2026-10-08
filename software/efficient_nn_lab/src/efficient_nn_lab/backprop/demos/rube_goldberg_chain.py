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

A bolinha e as engenhocas são uma simulação física REAL (pymunk --
`rube_goldberg_physics.py`), não uma animação desenhada à mão: a bolinha
rola pelas rampas seguindo a inclinação real delas, a alavanca tomba de
verdade sob o impacto (presa por um pivô com mola, não um ângulo
escolhido a dedo), e a roldana gira por fricção de contato -- a física
decide a trajetória, esta demo só decide O QUE É MOSTRADO em cada trecho
dela. A física nunca lê `self.target`: ela é pura mecânica (rampas, uma
alavanca pivotada, um disco livre, gravidade), igual em toda chamada --
só os números escritos nos chips (σ', w2, x...) dependem do alvo.

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

Números fixos e determinísticos (ESPECIFICACAO_DLVL.md #35) -- inclusive a
simulação física, que não tem nenhuma fonte de aleatoriedade (posição e
velocidade iniciais fixas, passo de tempo fixo).
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.backprop.demos.chain_rule_layers import compute_chain_1_1_1
from efficient_nn_lab.backprop.demos.rube_goldberg_physics import (
    LEG_X_BOUNDS,
    final_sample,
    leg_boundary_sample,
    leg_samples,
)
from efficient_nn_lab.core.demo import DemoModule, Frame, slider, transition

#: Quantos elos (= engenhocas) a cadeia tem nesta rede 1->1->1 -- usado para
#: dimensionar os arrays one-hot `station_fired`/`station_glow`. O renderer
#: (widgets/renderers/rube_goldberg.py) não importa esta constante: ele
#: deriva o mesmo número de `len(chain_values)`, mantendo física (posições)
#: e semântica (quantos elos) como dois donos independentes que só
#: concordam porque os dois contam o mesmo tuple.
N_STATIONS = 5

#: Quantos frames de simulação física populam cada transição entre
#: engenhocas -- mesma ordem de grandeza do DEFAULT_TWEEN_STEPS genérico
#: (core/demo.py), mas escolhido aqui porque é a física, não o easing
#: genérico, que decide o movimento.
PHYSICS_TWEEN_STEPS = 10

#: Os cinco fatores de ∂L/∂w1, na ordem em que a cadeia os concatena (da
#: perda para o peso) -- mesma ordem e mesmos nomes de chain_rule_layers.py,
#: para as duas demos lerem como "a mesma cadeia, duas peles".
CHAIN_NAMES = ("∂L/∂a2", "σ'(z2)", "w2", "σ'(z1)", "x")


def _ball_fields(sample) -> dict[str, object]:
    return {
        "ball_x": sample.ball_x, "ball_y": sample.ball_y, "ball_angle": sample.ball_angle,
        "lever_angle": sample.lever_angle, "pulley_angle": sample.pulley_angle,
    }


class RubeGoldbergChainDemo(DemoModule):
    title = "Backprop -> A regra da cadeia como máquina de Rube Goldberg"
    slug = "backprop.rube_goldberg"
    description = (
        "A mesma rede 1->1->1 e os mesmos cinco fatores de ∂L/∂w1 da demo "
        "'Camadas e a regra da cadeia', agora como uma máquina de verdade (física real, "
        "pymunk): uma bolinha nasce do erro da rede, rola pelas rampas seguindo a inclinação "
        "delas, tomba a alavanca sob o próprio peso e gira a roldana por fricção -- uma "
        "engenhoca por fator, até cair no balde final com ∂L/∂w1. Clique na tela para "
        "empurrar a bolinha para a próxima engenhoca."
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
        spawn = _ball_fields(leg_boundary_sample(0, end=False))

        checkpoints: list[Frame] = []

        def snap(label: str, explanation: str, equation: str, **overrides: object) -> None:
            values: dict[str, object] = {
                "kind": "rube_goldberg",
                "target": self.target, "loss": c["loss"],
                "chain_names": CHAIN_NAMES, "chain_values": chain_values,
                "chain_partials": chain_partials, "g_w1": chain_partials[-1],
                "ball_value": 0.0, "ball_reveal": 0.0, "bucket_reveal": 0.0,
                "station_fired": no_fired, "station_glow": no_glow,
                **spawn,
            }
            values.update(overrides)
            checkpoints.append(Frame(label, values, explanation, equation))

        snap(
            "A máquina",
            "Esta é a mesma regra da cadeia da demo 'Camadas e a regra da cadeia', só que em "
            "forma de máquina -- com física real: uma bolinha nasce do erro da rede e rola de "
            "verdade por cinco engenhocas, uma por fator da cadeia, cada uma multiplicando a "
            "bolinha pelo SEU fator -- exatamente como cada bloco daquela outra demo. Clique em "
            "qualquer lugar da tela (ou em 'Próximo') para dar o primeiro empurrão.",
            "∂L/∂w1 = ∂L/∂a2 · σ'(z_2) · w_2 · σ'(z_1) · x",
        )
        snap(
            "A bolinha cai no funil",
            f"O funil recebe o erro da rede: ∂L/∂a2 = {chain_values[0]:+.4f}. É o primeiro fator "
            "da cadeia -- a bolinha nasce com este valor e cai pelo funil até o topo da primeira "
            "rampa.",
            "∂L/∂a2 = a_2 - alvo",
            **_ball_fields(leg_boundary_sample(0, end=True)),
            ball_value=chain_partials[0], ball_reveal=1.0,
            station_fired=onehot(0), station_glow=onehot(0),
        )
        snap(
            "Desce a rampa da camada 2",
            f"A rampa multiplica pela derivada da ativação, σ'(z2) = {chain_values[1]:.4f} -- "
            "sempre um número entre 0 e 0,25, então a bolinha SEMPRE encolhe aqui. Produto até "
            f"agora: {chain_partials[0]:+.4f} · {chain_values[1]:.4f} = {chain_partials[1]:+.5f}.",
            "σ'(z_2) = a_2 (1 - a_2)",
            **_ball_fields(leg_boundary_sample(1, end=True)),
            ball_value=chain_partials[1], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1)), station_glow=onehot(1),
        )
        sign_note = (
            "Como w2 é negativo, o número que a alavanca aplica inverte o sinal da bolinha -- "
            "ela sai com a cor trocada (não é a física que inverte, é o PESO: a alavanca sempre "
            "tomba e lança a bolinha para a frente, só o número que ela multiplica é que pode "
            "ser negativo)."
            if chain_values[2] < 0
            else "w2 é positivo aqui, então a alavanca não troca o sinal da bolinha."
        )
        snap(
            "A alavanca do peso w2",
            f"A alavanca multiplica pelo peso w2 = {chain_values[2]:+.2f}. {sign_note} Produto "
            f"até agora: {chain_partials[2]:+.5f}.",
            "∂z2/∂a1 = w_2",
            **_ball_fields(leg_boundary_sample(2, end=True)),
            ball_value=chain_partials[2], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2)), station_glow=onehot(2),
        )
        snap(
            "Desce a rampa da camada 1",
            f"Segunda rampa, derivada local da SUA PRÓPRIA ativação: σ'(z1) = {chain_values[3]:.4f}. "
            "Não é a mesma rampa da camada 2 -- cada ativação tem a sua. Produto até agora: "
            f"{chain_partials[3]:+.6f}.",
            "σ'(z_1) = a_1 (1 - a_1)",
            **_ball_fields(leg_boundary_sample(3, end=True)),
            ball_value=chain_partials[3], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3)), station_glow=onehot(3),
        )
        snap(
            "A roldana da entrada",
            "Última engenhoca: multiplica pela entrada que passou por w1, x = "
            f"{chain_values[4]:+.2f}. É o quinto e último fator -- produto final: "
            f"{chain_partials[4]:+.6f}.",
            "∂z1/∂w1 = x",
            **_ball_fields(leg_boundary_sample(4, end=True)),
            ball_value=chain_partials[4], ball_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3) + onehot(4)),
            station_glow=onehot(4),
        )
        bucket = final_sample()
        snap(
            "O balde final: ∂L/∂w1",
            f"A bolinha cai no balde com o produto dos cinco fatores: ∂L/∂w1 = "
            f"{chain_partials[4]:+.6f}. É exatamente o mesmo número que a demo 'Camadas e a "
            "regra da cadeia' calcula bloco por bloco -- duas máquinas, um resultado.",
            "∂L/∂w1 = δ_1 · x",
            **_ball_fields(bucket),
            ball_value=chain_partials[4], ball_reveal=1.0, bucket_reveal=1.0,
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
            **_ball_fields(bucket),
            ball_value=chain_partials[4], ball_reveal=1.0, bucket_reveal=1.0,
            station_fired=(onehot(0) + onehot(1) + onehot(2) + onehot(3) + onehot(4)),
            station_glow=all_glow,
        )

        return self._interleave_with_physics(checkpoints)

    @staticmethod
    def _interleave_with_physics(checkpoints: list[Frame]) -> list[Frame]:
        """Like `core.demo.build_sequence`, but the gaps that correspond to
        physical motion get their ball/lever/pulley fields from the real
        simulated trajectory instead of a generic eased interpolation.

        `checkpoints` is always [intro, funnel, ramp1, lever, ramp2, pulley,
        bucket, recap] (8 entries, built in `_build_frames`): the first 6
        gaps (intro->funnel, funnel->ramp1, ..., pulley->bucket) are where
        the ball actually moves, in the same order as `LEG_X_BOUNDS` --
        EVERY click animates real motion, including the first one (the
        ball visibly falls through the funnel, not just fades in in
        place). Only the last gap (bucket->recap, a glow-only change with
        the ball already at rest) has no motion and keeps the ordinary
        generic tween untouched.
        """
        n_motion_gaps = len(LEG_X_BOUNDS)
        sequence = [checkpoints[0]]
        for gap_index, (a, b) in enumerate(zip(checkpoints, checkpoints[1:])):
            tweened = transition(a, b, steps=PHYSICS_TWEEN_STEPS)
            leg_index = gap_index
            if 0 <= leg_index < n_motion_gaps:
                samples = leg_samples(leg_index, PHYSICS_TWEEN_STEPS)
                for frame, sample in zip(tweened, samples):
                    frame.values.update(_ball_fields(sample))
            sequence.extend(tweened)
            sequence.append(b)
        return sequence
