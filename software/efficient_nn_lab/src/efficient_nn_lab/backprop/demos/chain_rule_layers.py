"""Demonstração — a correspondência entre as camadas e a regra da cadeia.

As outras demos de backprop mostram a mecânica (`backprop.classic`), a rede
inteira neurônio por neurônio (`backprop.mlp`) e a forma matricial
(`backprop.matrix`). Falta a pergunta que o aluno faz depois de ver as
três: **de onde exatamente sai cada fator daquele produto?** A regra da
cadeia costuma ser apresentada como uma fila de derivadas que "vem da
matemática", sem endereço na rede -- e aí o aluno decora a fórmula sem
saber por que ela tem cinco fatores e não três, nem por que o viés não
aparece multiplicando nada.

Esta demo existe para dar endereço a cada fator. A ideia central:

* cada camada não é UM bloco, são **dois**: a operação linear
  (`z = w·entrada + b`) e a ativação (`a = σ(z)`). A rede aparece aqui como
  seis blocos em fila -- `x`, `z1`, `a1`, `z2`, `a2`, `L` -- justamente
  porque a regra da cadeia trata cada um como um elo separado;
* embaixo de cada bloco fica a **derivada local daquele bloco**, alinhada
  verticalmente com ele. Essa é a correspondência: um bloco, um fator. Onde
  o bloco tem parâmetros, ele tem uma derivada local por parâmetro -- e
  todas aparecem, nenhuma fica implícita;
* o produto acumulado (`δ`) desce a fila da direita para a esquerda, e cada
  parâmetro pega o `δ` da sua camada e multiplica pela sua própria derivada
  local. É isso que "concatenar" quer dizer, e é por isso que
  `∂L/∂b = δ`: a derivada local do viés vale exatamente 1.

Rede 1 → 1 → 1 de propósito: com um neurônio por camada o caminho de cada
peso até a perda é **único**, então a regra da cadeia é um produto puro,
sem soma sobre caminhos (a demo de 4 camadas cobre o caso com vários
caminhos). Assim os cinco fatores de ∂L/∂w1 caberem na tela em corpo
grande, um por bloco.

**Um passo = um número novo.** Cada peso, cada viés, cada `z`, cada
ativação, cada derivada local, cada `δ` e cada gradiente têm o seu próprio
passo -- inclusive as derivadas locais dos vieses, que valem 1 e cuja
tentação é omitir "porque não mudam nada". Omiti-las é exatamente o que faz
o aluno não entender de onde vem `∂L/∂b`. Verificado por teste
(`test_chain_demo_reveals_at_most_one_new_value_per_step`).

Números fixos e determinísticos (ESPECIFICACAO_DLVL.md #35).
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.backprop.activation import sigmoid, sigmoid_derivative
from efficient_nn_lab.bitnet.linear import loss_gradient_wrt_y, squared_error_loss
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence

#: Rede 1 -> 1 -> 1, sigmoide nas duas camadas, viés em ambas.
#:
#: Escolhidos para que a demonstração tenha as três propriedades que a
#: tornam legível:
#:
#: * nenhum fator da cadeia vale 1,00 por acidente (um fator 1,00 deixaria o
#:   produto parcial idêntico antes e depois daquele passo, e o passo
#:   pareceria não ter feito nada). Os ÚNICOS 1,00 são as derivadas locais
#:   dos vieses -- e ali o 1,00 é o conteúdo do passo, não um acidente;
#: * `w2` é **negativo**, então o gradiente troca de sinal ao atravessar a
#:   camada 2. É o passo que mostra que o backward não é só "encolher
#:   números": o sinal do peso decide para que lado a camada anterior é
#:   corrigida;
#: * nenhuma sigmoide satura (todos os z ficam bem dentro de [-1, 1]), então
#:   nenhuma derivada local sai perto de zero e o produto não colapsa.
_X = 0.9
_W1 = 0.8
_B1 = -0.2
_W2 = -1.3
_B2 = 0.4

#: Os cinco fatores de ∂L/∂w1, na ordem em que a cadeia os concatena
#: (da perda para o peso). Um por bloco do desenho de cima.
_N_CHAIN_FACTORS = 5

#: Todo campo que revela UM número na tela. A regra "um número novo por
#: passo" é enunciada em cima desta lista e verificada sobre ela em
#: tests/test_backprop.py -- ao acrescentar uma quantidade, acrescente aqui.
VALUE_REVEAL_FIELDS = (
    "rv_x", "rv_w1", "rv_b1", "rv_z1", "rv_a1",
    "rv_w2", "rv_b2", "rv_z2", "rv_a2", "rv_target", "rv_loss",
    "rv_dL_da2", "rv_sp2", "rv_delta2",
    "rv_dz2_dw2", "rv_g_w2", "rv_dz2_db2", "rv_g_b2",
    "rv_dz2_da1", "rv_dL_da1", "rv_sp1", "rv_delta1",
    "rv_dz1_dw1", "rv_g_w1", "rv_dz1_db1", "rv_g_b1",
    "rv_chain",
)

#: Índices dos seis blocos da fila de cima, na ordem do forward. Usados
#: tanto pelo módulo (para acender o bloco do passo) quanto pelo renderer
#: (para posicioná-los) -- uma única fonte da verdade para a ordem.
NODE_X, NODE_Z1, NODE_A1, NODE_Z2, NODE_A2, NODE_L = range(6)
_N_NODES = 6
#: Parâmetros, na ordem em que aparecem embaixo dos blocos.
PARAM_W1, PARAM_B1, PARAM_W2, PARAM_B2, PARAM_TARGET = range(5)
_N_PARAMS = 5
#: Slots de derivada local: (bloco, linha). A linha 0 é a "estrada" do
#: backward -- a que continua para a esquerda; as linhas 1 e 2 são as
#: ramificações para os parâmetros daquele bloco.
_N_CARD_ROWS = 3


#: De qual bloco cada fator da cadeia é a derivada local -- dito em palavras
#: no texto de cada passo, para o fator nunca aparecer sem endereço.
_FACTOR_SOURCE = (
    "o bloco da perda L",
    "a ativação da camada 2",
    "a operação linear da camada 2",
    "a ativação da camada 1",
    "a operação linear da camada 1",
)
#: E qual bloco acender no desenho de cima enquanto aquele fator entra.
_FACTOR_NODE = (NODE_L, NODE_A2, NODE_Z2, NODE_A1, NODE_Z1)


def _copy_state(state: dict[str, object]) -> dict[str, object]:
    """Snapshot do estado de um frame, com os arrays copiados.

    O construtor abaixo muta um único dicionário ``state`` em vez de
    reescrever cinquenta campos por checkpoint. Cada Frame precisa, então,
    da SUA cópia dos arrays mutáveis -- sem isso todos os frames
    compartilhariam (e mostrariam) o estado final de revelação.
    """
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


class ChainRuleLayersDemo(DemoModule):
    title = "Backprop -> Camadas e a regra da cadeia"
    slug = "backprop.chain"
    description = (
        "De onde sai cada fator da regra da cadeia? Cada camada aparece como dois blocos "
        "-- operação linear (w·entrada + b) e ativação (σ) -- e embaixo de cada bloco fica a "
        "sua derivada local, alinhada com ele: um bloco, um fator. O produto acumulado δ "
        "desce a fila da direita para a esquerda e cada parâmetro (w1, b1, w2, b2) pega o δ "
        "da sua camada e multiplica pela sua própria derivada local. Nenhum parâmetro fica "
        "de fora, inclusive as derivadas dos vieses, que valem 1 -- é daí que sai ∂L/∂b = δ."
    )

    def __init__(self) -> None:
        self.target = 0.2
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "target": {"label": "Alvo (0-1)", "min": 0.05, "max": 0.95, "step": 0.05, "value": self.target},
        }

    # -- a conta: um forward e um backward completos -----------------------
    def _forward_backward(self) -> dict[str, float]:
        """Forward e backward desta rede, com TODOS os fatores intermediários.

        Devolve cada derivada local em separado (não só os gradientes
        finais), porque cada uma é um passo próprio da animação -- e porque
        é sobre elas que os testes conferem a cadeia.
        """
        x, target = _X, self.target

        z1 = _W1 * x + _B1
        a1 = float(sigmoid(z1))
        z2 = _W2 * a1 + _B2
        a2 = float(sigmoid(z2))
        loss = float(squared_error_loss(a2, target))

        # derivadas locais, uma por elo do caminho
        dL_da2 = float(loss_gradient_wrt_y(a2, target))  # a2 - alvo
        sp2 = float(sigmoid_derivative(z2))              # ∂a2/∂z2
        dz2_da1 = _W2                                    # ∂z2/∂a1
        sp1 = float(sigmoid_derivative(z1))              # ∂a1/∂z1
        dz2_dw2 = a1                                     # ∂z2/∂w2
        dz1_dw1 = x                                      # ∂z1/∂w1
        # ∂z/∂b = 1 para qualquer camada: o viés entra somando, com
        # coeficiente 1. É o único fator que vale exatamente 1 -- e é por
        # isso que o gradiente do viés é o próprio δ.
        dz2_db2 = 1.0
        dz1_db1 = 1.0

        # produtos acumulados: δ é o gradiente do lado de dentro da ativação
        delta2 = dL_da2 * sp2
        dL_da1 = delta2 * dz2_da1
        delta1 = dL_da1 * sp1

        return {
            "x": x, "w1": _W1, "b1": _B1, "z1": z1, "a1": a1,
            "w2": _W2, "b2": _B2, "z2": z2, "a2": a2,
            "target": target, "loss": loss,
            "dL_da2": dL_da2, "sp2": sp2, "delta2": delta2,
            "dz2_dw2": dz2_dw2, "g_w2": delta2 * dz2_dw2,
            "dz2_db2": dz2_db2, "g_b2": delta2 * dz2_db2,
            "dz2_da1": dz2_da1, "dL_da1": dL_da1,
            "sp1": sp1, "delta1": delta1,
            "dz1_dw1": dz1_dw1, "g_w1": delta1 * dz1_dw1,
            "dz1_db1": dz1_db1, "g_b1": delta1 * dz1_db1,
        }

    def _build_frames(self) -> list[Frame]:  # noqa: PLR0915 - one call per checkpoint
        c = self._forward_backward()
        x, w1, b1, z1, a1 = c["x"], c["w1"], c["b1"], c["z1"], c["a1"]
        w2, b2, z2, a2 = c["w2"], c["b2"], c["z2"], c["a2"]
        target, loss = c["target"], c["loss"]
        dL_da2, sp2, delta2 = c["dL_da2"], c["sp2"], c["delta2"]
        dz2_dw2, g_w2, dz2_db2, g_b2 = c["dz2_dw2"], c["g_w2"], c["dz2_db2"], c["g_b2"]
        dz2_da1, dL_da1, sp1, delta1 = c["dz2_da1"], c["dL_da1"], c["sp1"], c["delta1"]
        dz1_dw1, g_w1, dz1_db1, g_b1 = c["dz1_dw1"], c["g_w1"], c["dz1_db1"], c["g_b1"]

        # os cinco fatores, na ordem da concatenação (perda -> peso)
        chain_names = ("∂L/∂a2", "σ'(z2)", "w2", "σ'(z1)", "x")
        chain_values = (dL_da2, sp2, dz2_da1, sp1, dz1_dw1)
        chain_partials = tuple(float(np.prod(chain_values[: i + 1])) for i in range(_N_CHAIN_FACTORS))
        assert len(chain_values) == _N_CHAIN_FACTORS

        state: dict[str, object] = {
            "kind": "chain_layers",
            # -- os números (constantes em todos os frames) ---------------
            "x": x, "w1": w1, "b1": b1, "z1": z1, "a1": a1,
            "w2": w2, "b2": b2, "z2": z2, "a2": a2,
            "target": target, "loss": loss,
            "dL_da2": dL_da2, "sp2": sp2, "delta2": delta2,
            "dz2_dw2": dz2_dw2, "g_w2": g_w2, "dz2_db2": dz2_db2, "g_b2": g_b2,
            "dz2_da1": dz2_da1, "dL_da1": dL_da1, "sp1": sp1, "delta1": delta1,
            "dz1_dw1": dz1_dw1, "g_w1": g_w1, "dz1_db1": dz1_db1, "g_b1": g_b1,
            "chain_names": chain_names, "chain_values": chain_values,
            # -- revelações: 0 = ainda não calculado, 1 = na tela. Números
            # de verdade (não booleanos) porque core/math_utils.tween_values
            # os interpola -- é isso que faz cada quantidade *aparecer*
            # gradualmente em vez de piscar.
            "rv_graph": 0.0,
            "rv_x": 0.0, "rv_w1": 0.0, "rv_b1": 0.0, "rv_z1": 0.0, "rv_a1": 0.0,
            "rv_w2": 0.0, "rv_b2": 0.0, "rv_z2": 0.0, "rv_a2": 0.0,
            "rv_target": 0.0, "rv_loss": 0.0,
            "rv_dL_da2": 0.0, "rv_sp2": 0.0, "rv_delta2": 0.0,
            "rv_dz2_dw2": 0.0, "rv_g_w2": 0.0, "rv_dz2_db2": 0.0, "rv_g_b2": 0.0,
            "rv_dz2_da1": 0.0, "rv_dL_da1": 0.0, "rv_sp1": 0.0, "rv_delta1": 0.0,
            "rv_dz1_dw1": 0.0, "rv_g_w1": 0.0, "rv_dz1_db1": 0.0, "rv_g_b1": 0.0,
            "rv_chain": np.zeros(_N_CHAIN_FACTORS),
            "rv_check": 0.0, "rv_summary": 0.0,
            # -- destaques: acendem o bloco/parâmetro/carta de que o passo
            # está falando. hl_cards[bloco, linha] acende a derivada local
            # E o bloco de cima ao mesmo tempo -- é a correspondência que
            # esta demo existe para tornar mecânica.
            "hl_nodes": np.zeros(_N_NODES),
            "hl_params": np.zeros(_N_PARAMS),
            "hl_cards": np.zeros((_N_NODES, _N_CARD_ROWS)),
            # -- produto parcial da cadeia (número interpolado, então ele
            # visivelmente anda durante a transição)
            "chain_product": 0.0, "rv_chain_product": 0.0,
            "phase": "forward",
            "work_text": "",
        }

        frames: list[Frame] = []

        def snap(label: str, explanation: str, equation: str = "", **updates: object) -> None:
            for key, value in updates.items():
                if key not in state:
                    # Sem fallback silencioso (CLAUDE.md): um nome de campo
                    # digitado errado acrescentaria um campo que nenhum
                    # renderer lê, e o passo simplesmente não animaria.
                    raise KeyError(f"campo de frame desconhecido: {key!r}")
                state[key] = value
            frames.append(Frame(label, _copy_state(state), explanation, equation))

        def node(index: int) -> np.ndarray:
            spot = np.zeros(_N_NODES)
            spot[index] = 1.0
            return spot

        def param(index: int) -> np.ndarray:
            spot = np.zeros(_N_PARAMS)
            spot[index] = 1.0
            return spot

        def card(index: int, row: int) -> np.ndarray:
            spot = np.zeros((_N_NODES, _N_CARD_ROWS))
            spot[index, row] = 1.0
            return spot

        no_nodes = np.zeros(_N_NODES)
        no_params = np.zeros(_N_PARAMS)
        no_cards = np.zeros((_N_NODES, _N_CARD_ROWS))

        # ============ fase 1: a estrutura, antes de qualquer número =======
        snap(
            "Uma camada são dois blocos",
            "Uma camada faz duas coisas diferentes, e a regra da cadeia trata cada uma como um "
            "elo separado: primeiro a operação linear (multiplica pelos pesos, soma o viés), "
            "depois a ativação (a não linearidade). Por isso esta rede 1→1→1 aparece como SEIS "
            "blocos e não como dois neurônios. Cada bloco vai ganhar, embaixo, a sua derivada "
            "local -- um bloco, um fator.",
            equation="z = w · entrada + b,   a = σ(z)",
            rv_graph=1.0,
            work_text=(
                "camada = [operação linear] + [ativação]\n"
                "  x -> z1 = w1·x + b1 -> a1 = σ(z1) -> z2 = w2·a1 + b2 -> a2 = σ(z2) -> L\n"
                "5 elos no caminho de w1 até L  =>  a cadeia de w1 terá 5 fatores."
            ),
        )

        # ============ fase 2: forward, um número por passo ================
        snap(
            "A entrada x",
            f"A entrada vale x = {x:.2f}. Não é parâmetro: vem dos dados e não será atualizada. "
            "Mesmo assim vai aparecer no backward, como derivada local do bloco z1 em relação a "
            "w1 -- guarde esse número.",
            equation="x",
            rv_x=1.0, hl_nodes=node(NODE_X),
            work_text=f"x = {x:+.2f}   (dado, não parâmetro)",
        )
        snap(
            "O peso w1",
            f"Primeiro parâmetro: w1 = {w1:+.2f}. Ele multiplica a entrada. A pergunta que o "
            "backward vai responder é exatamente esta: se w1 mudasse um pouquinho, quanto L "
            "mudaria?",
            equation="w_1",
            rv_w1=1.0, hl_nodes=no_nodes, hl_params=param(PARAM_W1),
            work_text=f"w1 = {w1:+.2f}   (parâmetro: será atualizado)",
        )
        snap(
            "O viés b1",
            f"Segundo parâmetro do mesmo bloco: b1 = {b1:+.2f}. O viés não multiplica nada -- ele "
            "só soma. Essa diferença parece pequena agora, mas é ela que vai dar ao viés a "
            "derivada local mais simples de toda a demo.",
            equation="b_1",
            rv_b1=1.0, hl_params=param(PARAM_B1),
            work_text=f"b1 = {b1:+.2f}   (parâmetro: soma, não multiplica)",
        )
        snap(
            "z1: a operação linear",
            f"O bloco linear da camada 1 combina os três: w1·x + b1 = {w1:+.2f}·{x:.2f} + "
            f"({b1:+.2f}) = {z1:+.4f}. Repare que é uma reta em w1 -- e a derivada de uma reta é "
            "o coeficiente, o que vai deixar o backward deste bloco trivial.",
            equation="z_1 = w_1 · x + b_1",
            rv_z1=1.0, hl_nodes=node(NODE_Z1), hl_params=no_params,
            work_text=f"z1 = w1·x + b1\n   = {w1:+.2f}·{x:.2f} + ({b1:+.2f})\n   = {z1:+.4f}",
        )
        snap(
            "a1: a ativação",
            f"Agora o segundo bloco da camada, que é um passo à parte: z1 = {z1:+.4f} entra na "
            f"sigmoide e sai a1 = {a1:.4f}. É aqui que a rede deixa de ser linear -- e é aqui "
            "que o backward vai precisar de uma derivada que não é constante.",
            equation="a_1 = σ(z_1)",
            rv_a1=1.0, hl_nodes=node(NODE_A1),
            work_text=f"a1 = σ({z1:+.4f}) = {a1:.4f}\nsem esta não linearidade, empilhar camadas não acrescentaria nada.",
        )
        snap(
            "O peso w2",
            f"A camada 2 repete a mesma estrutura com os seus próprios parâmetros: w2 = {w2:+.2f}. "
            "Negativo, de propósito -- mais adiante ele vai inverter o sinal do gradiente que "
            "volta para a camada 1.",
            equation="w_2",
            rv_w2=1.0, hl_nodes=no_nodes, hl_params=param(PARAM_W2),
            work_text=f"w2 = {w2:+.2f}   (negativo: vai trocar o sinal no backward)",
        )
        snap(
            "O viés b2",
            f"E b2 = {b2:+.2f}. Quatro parâmetros no total nesta rede: w1, b1, w2, b2. Ao final "
            "cada um vai ter o seu gradiente, calculado pela mesma receita.",
            equation="b_2",
            rv_b2=1.0, hl_params=param(PARAM_B2),
            work_text=f"b2 = {b2:+.2f}\nparâmetros da rede: w1, b1, w2, b2  (quatro gradientes ao final)",
        )
        snap(
            "z2: a operação linear da camada 2",
            f"Mesma conta, entrada diferente: agora quem entra é a1 (a saída da camada anterior), "
            f"não x. z2 = {w2:+.2f}·{a1:.4f} + {b2:+.2f} = {z2:+.4f}. Essa troca de entrada é o "
            "que encadeia as camadas.",
            equation="z_2 = w_2 · a_1 + b_2",
            rv_z2=1.0, hl_nodes=node(NODE_Z2), hl_params=no_params,
            work_text=f"z2 = w2·a1 + b2\n   = {w2:+.2f}·{a1:.4f} + ({b2:+.2f})\n   = {z2:+.4f}",
        )
        snap(
            "a2: a saída da rede",
            f"A ativação da camada 2 dá a resposta da rede: a2 = σ({z2:+.4f}) = {a2:.4f}. "
            "Chegamos ao fim do forward -- seis blocos, seis quantidades.",
            equation="a_2 = σ(z_2)",
            rv_a2=1.0, hl_nodes=node(NODE_A2),
            work_text=f"a2 = σ({z2:+.4f}) = {a2:.4f}   (a resposta da rede)",
        )
        snap(
            "O alvo",
            f"O alvo é {target:.2f}: vem dos dados, não da rede. Sozinho ainda não é erro nenhum "
            "-- é só a referência contra a qual a resposta vai ser medida.",
            equation="alvo",
            rv_target=1.0, hl_nodes=no_nodes, hl_params=param(PARAM_TARGET),
            work_text=f"a2   = {a2:.4f}   (o que a rede deu)\nalvo = {target:.2f}     (o que se queria)",
        )
        snap(
            "L: a perda fecha a cadeia",
            f"A perda transforma a diferença num único número: L = ½·({a2:.4f} − {target:.2f})² = "
            f"{loss:.4f}. Ela é o último bloco da fila, e é a partir dela que o backward começa a "
            "andar para trás -- derivar um escalar é o que dá sentido a 'gradiente'.",
            equation="L = ½ (a_2 - alvo)^2",
            rv_loss=1.0, hl_nodes=node(NODE_L), hl_params=no_params,
            work_text=f"L = ½ · ({a2 - target:+.4f})²\n  = {loss:.4f}",
        )

        # ============ fase 3: derivadas locais + concatenação ============
        snap(
            "∂L/∂a2: a derivada do último bloco",
            f"O backward começa no fim e pergunta, bloco por bloco: 'se a MINHA saída subisse um "
            f"pouquinho, quanto L subiria?'. Para o erro quadrático a resposta é a própria "
            f"diferença: ∂L/∂a2 = {a2:.4f} − {target:.2f} = {dL_da2:+.4f}. Primeira carta na fila "
            "de baixo, alinhada com o bloco L.",
            equation="∂L/∂a2 = a_2 - alvo",
            phase="backward",
            rv_dL_da2=1.0, hl_nodes=node(NODE_L), hl_cards=card(NODE_L, 0),
            work_text=f"∂L/∂a2 = a2 - alvo = {dL_da2:+.4f}\n(derivada local do bloco L em relação à sua entrada, a2)",
        )
        snap(
            "∂a2/∂z2 = σ'(z2)",
            f"Próximo bloco à esquerda é a ativação da camada 2. A derivada local dela não é "
            f"constante: σ'(z2) = a2·(1 − a2) = {a2:.4f}·{1 - a2:.4f} = {sp2:.4f}. Este passo só "
            "CALCULA a derivada -- ainda não multiplicou nada por ela.",
            equation="∂a2/∂z2 = σ'(z_2) = a_2 (1 - a_2)",
            rv_sp2=1.0, hl_nodes=node(NODE_A2), hl_cards=card(NODE_A2, 0),
            work_text=f"σ'(z2) = a2·(1 - a2)\n       = {a2:.4f} · {1 - a2:.4f}\n       = {sp2:.4f}",
        )
        snap(
            "δ2: os dois primeiros fatores, concatenados",
            f"Agora a multiplicação: {dL_da2:+.4f} · {sp2:.4f} = {delta2:+.5f}. Este produto tem "
            "nome, δ2, e endereço: é o gradiente do lado de DENTRO da ativação da camada 2, ou "
            "seja ∂L/∂z2. Guardá-lo é o que evita recalcular a mesma cadeia para cada parâmetro "
            "da camada.",
            equation="δ_2 = ∂L/∂z2 = ∂L/∂a2 · σ'(z_2)",
            rv_delta2=1.0, hl_nodes=node(NODE_Z2), hl_cards=no_cards,
            work_text=f"δ2 = ∂L/∂a2 · σ'(z2)\n   = {dL_da2:+.4f} · {sp2:.4f}\n   = {delta2:+.5f}   (= ∂L/∂z2)",
        )
        snap(
            "∂z2/∂w2 = a1",
            f"O bloco z2 tem três entradas -- w2, b2 e a1 -- e portanto TRÊS derivadas locais, "
            f"uma para cada. A do peso é o valor que entrou por ele: ∂z2/∂w2 = a1 = {a1:.4f}. "
            "Sempre: derivada em relação a um peso = a entrada daquele peso.",
            equation="∂z2/∂w2 = a_1",
            rv_dz2_dw2=1.0, hl_nodes=node(NODE_Z2), hl_cards=card(NODE_Z2, 1),
            hl_params=param(PARAM_W2),
            work_text=f"z2 = w2·a1 + b2\n∂z2/∂w2 = a1 = {a1:.4f}   (derivada de uma reta = o coeficiente)",
        )
        snap(
            "∂L/∂w2 = δ2 · a1",
            f"Primeiro gradiente pronto: o δ da camada vezes a derivada local do peso, "
            f"{delta2:+.5f} · {a1:.4f} = {g_w2:+.5f}. Note que a cadeia de w2 tem TRÊS fatores "
            "(∂L/∂a2, σ'(z2), a1), porque w2 está a três elos da perda.",
            equation="∂L/∂w2 = δ_2 · ∂z2/∂w2",
            rv_g_w2=1.0, hl_params=param(PARAM_W2), hl_cards=no_cards,
            work_text=f"∂L/∂w2 = δ2 · a1\n       = {delta2:+.5f} · {a1:.4f}\n       = {g_w2:+.5f}",
        )
        snap(
            "∂z2/∂b2 = 1",
            "Segunda derivada local do mesmo bloco, agora em relação ao viés. b2 entra somando, "
            "com coeficiente 1, então ∂z2/∂b2 = 1 exatamente. É o único fator desta demo que vale "
            "1 -- e é o conteúdo do passo, não um acidente: pular este passo é o que faz o aluno "
            "não saber de onde vem o gradiente do viés.",
            equation="∂z2/∂b2 = 1",
            rv_dz2_db2=1.0, hl_nodes=node(NODE_Z2), hl_cards=card(NODE_Z2, 2),
            hl_params=param(PARAM_B2),
            work_text="z2 = w2·a1 + b2\n∂z2/∂b2 = 1   (b2 soma com coeficiente 1)",
        )
        snap(
            "∂L/∂b2 = δ2 · 1 = δ2",
            f"Mesma receita, e o 1 não muda nada: ∂L/∂b2 = {delta2:+.5f}. Aqui está a razão de "
            "'o gradiente do viés é o δ da camada': não é uma regra à parte que se decora, é a "
            "regra da cadeia com um fator que vale 1.",
            equation="∂L/∂b2 = δ_2",
            rv_g_b2=1.0, hl_params=param(PARAM_B2), hl_cards=no_cards,
            work_text=f"∂L/∂b2 = δ2 · 1 = {g_b2:+.5f}\n=> gradiente do viés = δ da camada, sempre",
        )
        snap(
            "∂z2/∂a1 = w2: o caminho de volta",
            f"Terceira derivada local do bloco z2, e a que continua a viagem: em relação à "
            f"ENTRADA a1. Vale o peso, ∂z2/∂a1 = w2 = {w2:+.2f}. O gradiente volta atravessando o "
            "mesmo peso que o forward usou -- não existe um segundo conjunto de pesos para o "
            "backward.",
            equation="∂z2/∂a1 = w_2",
            rv_dz2_da1=1.0, hl_nodes=node(NODE_Z2), hl_cards=card(NODE_Z2, 0),
            hl_params=param(PARAM_W2),
            work_text=f"∂z2/∂a1 = w2 = {w2:+.2f}\no backward reusa os pesos do forward, na direção contrária.",
        )
        snap(
            "∂L/∂a1: o sinal se inverte",
            f"Concatenando: {delta2:+.5f} · {w2:+.2f} = {dL_da1:+.5f}. O sinal TROCOU, porque w2 é "
            "negativo. O backward não é só 'encolher números': o sinal do peso decide para que "
            "lado a camada anterior precisa ser corrigida.",
            equation="∂L/∂a1 = δ_2 · w_2",
            rv_dL_da1=1.0, hl_nodes=node(NODE_A1), hl_cards=no_cards, hl_params=no_params,
            work_text=f"∂L/∂a1 = δ2 · w2\n       = {delta2:+.5f} · {w2:+.2f}\n       = {dL_da1:+.5f}   (sinal invertido por w2 < 0)",
        )
        snap(
            "∂a1/∂z1 = σ'(z1)",
            f"A camada 1 tem a SUA ativação, com a SUA derivada local, calculada do seu próprio "
            f"a1: σ'(z1) = {a1:.4f}·{1 - a1:.4f} = {sp1:.4f}. Não é a mesma de cima -- cada "
            "sigmoide tem a sua, e é aí que gradientes desaparecem em redes profundas: basta "
            "multiplicar muitos números menores que 1.",
            equation="∂a1/∂z1 = σ'(z_1) = a_1 (1 - a_1)",
            rv_sp1=1.0, hl_nodes=node(NODE_A1), hl_cards=card(NODE_A1, 0),
            work_text=f"σ'(z1) = {a1:.4f} · {1 - a1:.4f} = {sp1:.4f}\n(cada ativação tem a sua; σ' ≤ 0,25 sempre)",
        )
        snap(
            "δ1: quatro fatores concatenados",
            f"O δ da camada 1: {dL_da1:+.5f} · {sp1:.4f} = {delta1:+.6f}. Ele já carrega quatro "
            "derivadas locais multiplicadas -- ∂L/∂a2, σ'(z2), w2 e σ'(z1) -- exatamente os "
            "quatro blocos à direita de z1. É isso que 'o gradiente se propaga' quer dizer.",
            equation="δ_1 = ∂L/∂z1 = ∂L/∂a1 · σ'(z_1)",
            rv_delta1=1.0, hl_nodes=node(NODE_Z1), hl_cards=no_cards,
            work_text=f"δ1 = ∂L/∂a1 · σ'(z1)\n   = {dL_da1:+.5f} · {sp1:.4f}\n   = {delta1:+.6f}   (= ∂L/∂z1, quatro fatores dentro)",
        )
        snap(
            "∂z1/∂w1 = x",
            f"Mesma regra do outro bloco linear, entrada diferente: a derivada local de z1 em "
            f"relação a w1 é o que entrou por w1, ou seja x = {x:.2f}. Em z2 esse papel era de "
            "a1; aqui é da entrada da rede.",
            equation="∂z1/∂w1 = x",
            rv_dz1_dw1=1.0, hl_nodes=node(NODE_Z1), hl_cards=card(NODE_Z1, 1),
            hl_params=param(PARAM_W1),
            work_text=f"z1 = w1·x + b1\n∂z1/∂w1 = x = {x:+.2f}",
        )
        snap(
            "∂L/∂w1 = δ1 · x",
            f"O gradiente do peso mais distante da perda: {delta1:+.6f} · {x:.2f} = {g_w1:+.6f}. "
            "Cinco elos, cinco fatores. E ainda assim a conta que acabou de ser feita tem só "
            "DUAS multiplicações -- porque δ1 já era o produto dos quatro primeiros.",
            equation="∂L/∂w1 = δ_1 · ∂z1/∂w1",
            rv_g_w1=1.0, hl_params=param(PARAM_W1), hl_cards=no_cards,
            work_text=f"∂L/∂w1 = δ1 · x\n       = {delta1:+.6f} · {x:+.2f}\n       = {g_w1:+.6f}",
        )
        snap(
            "∂z1/∂b1 = 1",
            "E a última derivada local que faltava: o viés da camada 1, que também entra somando. "
            "∂z1/∂b1 = 1, pelo mesmo motivo de b2. Quatro parâmetros, quatro derivadas locais -- "
            "nenhuma implícita.",
            equation="∂z1/∂b1 = 1",
            rv_dz1_db1=1.0, hl_nodes=node(NODE_Z1), hl_cards=card(NODE_Z1, 2),
            hl_params=param(PARAM_B1),
            work_text="z1 = w1·x + b1\n∂z1/∂b1 = 1",
        )
        snap(
            "∂L/∂b1 = δ1",
            f"Fecha os quatro gradientes: ∂L/∂b1 = {g_b1:+.6f}, o próprio δ1. Compare a fileira: "
            "os dois vieses receberam o δ da sua camada sem alteração; os dois pesos receberam o "
            "δ multiplicado pela entrada que passou por eles.",
            equation="∂L/∂b1 = δ_1",
            rv_g_b1=1.0, hl_params=param(PARAM_B1), hl_cards=no_cards,
            work_text=(
                f"∂L/∂w1 = δ1 · x  = {g_w1:+.6f}      ∂L/∂b1 = δ1 = {g_b1:+.6f}\n"
                f"∂L/∂w2 = δ2 · a1 = {g_w2:+.6f}      ∂L/∂b2 = δ2 = {g_b2:+.6f}\n"
                "peso: δ · (o que entrou por ele)   |   viés: δ · 1"
            ),
        )

        # ============ fase 4: a cadeia inteira, fator por fator ==========
        snap(
            "A cadeia de w1, elo por elo",
            "Os cinco fatores de ∂L/∂w1 estavam espalhados pelos blocos. Agora eles entram em "
            "fila na faixa de baixo, na ordem em que a cadeia os multiplica -- da perda até o "
            "peso. Cada um já apareceu como derivada local de um bloco; nenhum é novo.",
            equation="∂L/∂w1 = ∂L/∂a2 · σ'(z_2) · w_2 · σ'(z_1) · x",
            phase="chain",
            hl_nodes=no_nodes, hl_params=param(PARAM_W1), hl_cards=no_cards,
            work_text="a fila de baixo = os mesmos fatores das cartas, na ordem do produto.",
        )
        for i, (name, value) in enumerate(zip(chain_names, chain_values)):
            reveal = np.zeros(_N_CHAIN_FACTORS)
            reveal[: i + 1] = 1.0
            # laid out ACROSS, not down: one line of names over one line of
            # numbers is the concatenation itself, and it also keeps the
            # work block at three lines -- five stacked factors ran off the
            # bottom of the canvas.
            names_line = "  ·  ".join(f"{chain_names[k]}" for k in range(i + 1))
            values_line = "  ·  ".join(f"{chain_values[k]:+.4f}" for k in range(i + 1))
            factor_lines = f"cadeia até aqui:  {names_line}\n                  {values_line}"
            previous = chain_partials[i - 1] if i else 1.0
            snap(
                f"Fator {i + 1}/{_N_CHAIN_FACTORS}: {name}",
                f"Entra {name} = {value:+.4f}, a derivada local de {_FACTOR_SOURCE[i]}. O produto "
                f"parcial vai de {previous:+.6f} para {chain_partials[i]:+.6f}. Multiplicar um "
                "fator por vez é literalmente o que o backward faz -- ele só não guarda a fila "
                "toda, guarda o produto acumulado.",
                equation="∂L/∂w1 = ∂L/∂a2 · σ'(z_2) · w_2 · σ'(z_1) · x",
                rv_chain=reveal,
                hl_nodes=node(_FACTOR_NODE[i]),
                chain_product=chain_partials[i], rv_chain_product=1.0,
                work_text=f"{factor_lines}\nproduto parcial = {chain_partials[i]:+.6f}",
            )
        check_diff = abs(chain_partials[-1] - g_w1)
        snap(
            "O mesmo número, pelos dois caminhos",
            f"O produto dos cinco fatores deu {chain_partials[-1]:+.6f}. O caminho de δ, que nunca "
            f"escreveu essa fila, já havia dado ∂L/∂w1 = {g_w1:+.6f}. São o mesmo número porque "
            "são a mesma conta: δ é a cadeia parcialmente multiplicada e guardada para reuso.",
            equation="∂L/∂w1 = δ_1 · x",
            rv_check=1.0, hl_nodes=no_nodes, hl_params=param(PARAM_W1),
            work_text=(
                f"cadeia fator por fator : {chain_partials[-1]:+.8f}\n"
                f"δ1 · x                 : {g_w1:+.8f}\n"
                f"diferença              : {check_diff:.1e}"
            ),
        )
        snap(
            "A regra geral: cada camada acrescenta dois fatores",
            "Olhando a tela inteira: ao atravessar uma camada para trás, o gradiente é "
            "multiplicado por exatamente dois fatores -- a derivada da ativação, σ'(z), e o peso, "
            "w. Nada mais. Uma camada a mais entre o peso e a perda significa dois fatores a mais "
            "no produto; se esses fatores são pequenos, o gradiente desaparece (vanishing "
            "gradient), e se são grandes, explode. A demo do BitNet/STE é o que acontece quando "
            "um desses fatores é ZERO.",
            equation="δ_camada = δ_seguinte · w_seguinte · σ'(z)",
            phase="summary",
            rv_summary=1.0, hl_nodes=no_nodes, hl_params=no_params,
            work_text=(
                "para trás, por camada:   δ  ->  δ · w · σ'(z)      (dois fatores por camada)\n"
                "para cada parâmetro:     ∂L/∂w = δ · entrada       ∂L/∂b = δ · 1\n"
                "nesta rede: 2 camadas => 4 fatores + 1 do peso = os 5 fatores de ∂L/∂w1."
            ),
        )

        return build_sequence(frames, steps=6)

