"""Comparação ANN x BitNet x SNN (ESPECIFICACAO_DLVL.md #21, #22, #23).

One persistent three-column table that *grows* a row at a time — every
previously revealed row stays on screen, so by the last checkpoint the
whole comparison is visible at once and self-explanatory without needing
to remember what a now-vanished earlier slide said. Closes on the one
caution the spec insists on making explicit: efficiency is not a free
property of an architecture; it depends on hardware, implementation,
memory, bandwidth, sparsity, algorithm and workload (#21).

Rows compare like with like: the "Pesos" row compares weights in all three
columns and the "Ativação" row compares activations. (An earlier
"Representação" row set ANN/BitNet *weights* against SNN *spikes*, which
are activations, and claimed an SNN keeps no continuous value -- but its
weights and membrane potential are continuous.) The outputs row feeds the
same weighted sum, x . w, to all three, as #23 asks.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.linear import quantized_forward
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif

_X = (2.0, 3.0)
_W = (0.8, 0.2)
_SNN_STEPS = 30

_ROW_ORDER = ["Pesos", "Ativação", "Domínio temporal", "Treinamento", "Operação principal"]
_REVEAL_KEYS = {
    "Pesos": "reveal_weights",
    "Ativação": "reveal_activation",
    "Domínio temporal": "reveal_domain",
    "Treinamento": "reveal_training",
    "Operação principal": "reveal_operation",
}


class AnnBitnetSnnComparisonDemo(DemoModule):
    title = "Comparação -> ANN x BitNet x SNN"
    slug = "comparison"
    description = "Mesma entrada conceitual, três formas de representar e operar sobre ela — uma tabela que cresce, não uma sequência de telas soltas."

    def _build_frames(self) -> list[Frame]:
        y_ann = sum(x * w for x, w in zip(_X, _W))
        bitnet_result = quantized_forward(_X, _W, DEFAULT_THRESHOLD)
        # The SAME conceptual input as the other two columns (#23): the
        # weighted sum x . w drives the LIF neuron as a constant current. R = 1
        # keeps the drive on the scale of the sum itself, so the neuron
        # answers with a rate instead of saturating at one spike per step.
        snn_params = LIFParams(tau=5.0, r=1.0, v_th=1.0)
        snn_trace = simulate_lif(constant_current(y_ann, _SNN_STEPS), snn_params)
        snn_spike_count = int(snn_trace.spikes.sum())

        table = {
            "Pesos": ("FP32/BF16 contínuos", "ternários {-1,0,+1}", "contínuos, qualquer precisão"),
            "Ativação": ("contínua", "quantizada (8 bits na b1.58)", "spikes binários (0/1)"),
            "Domínio temporal": ("normalmente ausente", "normalmente ausente", "explícito"),
            "Treinamento": ("backprop direto", "backprop + STE", "backprop + surrogate gradient"),
            "Operação principal": ("MAC (multiply-accumulate)", "soma/subtração de baixa precisão", "somar o peso a cada spike"),
        }

        base = {
            "kind": "comparison_pipeline",
            "table_rows": _ROW_ORDER,
            "table": table,
            # row name -> reveal field, carried in the frame so the renderer
            # never keeps its own copy of the row names (a copy that drifted
            # is what a row rename would otherwise break, silently).
            "row_reveal_keys": dict(_REVEAL_KEYS),
            "reveal_weights": 0.0,
            "reveal_activation": 0.0,
            "reveal_domain": 0.0,
            "reveal_training": 0.0,
            "reveal_operation": 0.0,
            "reveal_outputs": 0.0,
            "y_ann": y_ann,
            "y_bitnet": bitnet_result.y,
            "snn_spike_count": snn_spike_count,
            "snn_steps": _SNN_STEPS,
            "reveal_gradients": 0.0,
            "reveal_caveat": 0.0,
        }

        def frame(label: str, explanation: str, **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation)

        # one real sentence per row, naming the actual cell values (from
        # `table` above) instead of a templated "see how they compare".
        row_narration = {
            "Pesos": (
                "Compare peso com peso. A ANN guarda cada peso em ponto flutuante (FP32/BF16); a "
                "BitNet reduz cada peso a {-1,0,+1}; a SNN usa pesos contínuos comuns, em qualquer "
                "precisão. O que é binário numa SNN não é o peso, é a ativação — próxima linha."
            ),
            "Ativação": (
                "Agora ativação com ativação: contínua na ANN; quantizada na BitNet (8 bits na "
                "b1.58); binária na SNN — o neurônio dispara (1) ou não (0). Mas o potencial de "
                "membrana que decide o disparo é contínuo: a SNN guarda valores contínuos, só não "
                "os transmite."
            ),
            "Domínio temporal": (
                "ANN e BitNet feedforward não têm tempo: uma passada, uma saída (redes recorrentes "
                "são a exceção). Na SNN o tempo é explícito — cada entrada vira uma série de spikes "
                "e a informação mora na distribuição deles no tempo."
            ),
            "Treinamento": (
                "Os três treinam com backprop. Na ANN toda função do caminho é diferenciável, e a "
                "derivada exata passa direto. Na BitNet a quantização do peso é um degrau, contornado "
                "pelo Straight-Through Estimator; na SNN o disparo é um degrau, contornado pelo "
                "gradiente substituto."
            ),
            "Operação principal": (
                "A operação dominante é o MAC de ponto flutuante na ANN; na BitNet vira "
                "soma/subtração de baixa precisão (sem multiplicação); na SNN, cada spike que chega "
                "soma o peso da sinapse ao potencial — acumulação, sem multiplicação, e só onde há "
                "spike."
            ),
        }

        # each checkpoint keeps every previous reveal at 1.0 and turns the
        # next one on — the table only ever grows.
        revealed: dict[str, float] = {}
        checkpoints: list[Frame] = []

        for row_name in _ROW_ORDER:
            revealed[_REVEAL_KEYS[row_name]] = 1.0
            checkpoints.append(frame(row_name, row_narration[row_name], **revealed))

        revealed["reveal_outputs"] = 1.0
        checkpoints.append(
            frame(
                "Saída, para a mesma entrada conceitual",
                (
                    f"A mesma soma ponderada nos três. ANN: $y = x · w = {y_ann:g}$ (contínuo). BitNet: "
                    f"$y = x · Q(w) = {bitnet_result.y:g}$ (pesos ternários). SNN: os mesmos ${y_ann:g}$ "
                    f"entram como corrente constante num LIF (R = {snn_params.r:g}, τ = {snn_params.tau:g}, "
                    f"V_th = {snn_params.v_th:g}), que responde com {snn_spike_count} spikes em "
                    f"{_SNN_STEPS} passos: a intensidade virou TAXA de disparo, não o mesmo número na "
                    "mesma escala."
                ),
                **revealed,
            )
        )

        revealed["reveal_gradients"] = 1.0
        checkpoints.append(
            frame(
                "Tipo de gradiente",
                "ANN: gradiente exato. BitNet: aproximado via Straight-Through Estimator. "
                "SNN: aproximado via gradiente substituto. Duas soluções semelhantes em espírito, não idênticas em fórmula.",
                **revealed,
            )
        )

        revealed["reveal_caveat"] = 1.0
        checkpoints.append(
            frame(
                "Advertência sobre eficiência",
                "Nenhum destes três garante eficiência energética só pela arquitetura: "
                "o ganho real depende de hardware, implementação, memória, largura de banda, "
                "esparsidade, algoritmo e workload.",
                **revealed,
            )
        )

        return build_sequence(checkpoints, steps=12)
