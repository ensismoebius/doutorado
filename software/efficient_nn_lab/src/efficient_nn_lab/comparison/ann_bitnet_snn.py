"""Comparação ANN x BitNet x SNN (ESPECIFICACAO_DLVL.md #21, #22, #23).

One persistent three-column table that *grows* a row at a time — every
previously revealed row stays on screen, so by the last checkpoint the
whole comparison is visible at once and self-explanatory without needing
to remember what a now-vanished earlier slide said. Closes on the one
caution the spec insists on making explicit: efficiency is not a free
property of an architecture; it depends on hardware, implementation,
memory, bandwidth, sparsity, algorithm and workload (#21, #669).
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.linear import quantized_forward
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif

_X = (2.0, 3.0)
_W = (0.8, 0.2)

_ROW_ORDER = ["Representação", "Ativação", "Domínio temporal", "Treinamento", "Operação principal"]
_REVEAL_KEYS = {
    "Representação": "reveal_repr",
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
        snn_trace = simulate_lif(constant_current(0.30, 30), LIFParams(tau=5.0, r=5.0, v_th=1.0))
        snn_spike_count = int(snn_trace.spikes.sum())

        table = {
            "Representação": ("FP32/BF16 continuo", "ternário {-1,0,+1}", "spikes (0/1) no tempo"),
            "Ativação": ("contínua", "quantizada conforme arquitetura", "spikes"),
            "Domínio temporal": ("normalmente implícito", "normalmente implícito", "explícito"),
            "Treinamento": ("backprop direto", "backprop + STE", "backprop + surrogate gradient"),
            "Operação principal": ("MAC (multiply-accumulate)", "soma/subtração de baixa precisão", "eventos/spikes"),
        }

        base = {
            "kind": "comparison_pipeline",
            "table_rows": _ROW_ORDER,
            "table": table,
            "reveal_repr": 0.0,
            "reveal_activation": 0.0,
            "reveal_domain": 0.0,
            "reveal_training": 0.0,
            "reveal_operation": 0.0,
            "reveal_outputs": 0.0,
            "y_ann": y_ann,
            "y_bitnet": bitnet_result.y,
            "snn_spike_count": snn_spike_count,
            "reveal_gradients": 0.0,
            "reveal_caveat": 0.0,
        }

        def frame(label: str, explanation: str, **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation)

        # each checkpoint keeps every previous reveal at 1.0 and turns the
        # next one on — the table only ever grows.
        revealed: dict[str, float] = {}
        checkpoints: list[Frame] = []

        for row_name in _ROW_ORDER:
            revealed[_REVEAL_KEYS[row_name]] = 1.0
            checkpoints.append(
                frame(row_name, f"{row_name}: veja como ANN, BitNet e SNN se comparam nesta linha.", **revealed)
            )

        revealed["reveal_outputs"] = 1.0
        checkpoints.append(
            frame(
                "Saída, para a mesma entrada conceitual",
                (
                    f"ANN: y = {y_ann:g} (contínuo). BitNet: y = {bitnet_result.y:g} (pesos ternários). "
                    f"SNN: {snn_spike_count} spikes em 30 passos, para uma corrente comparável — "
                    "uma codificação por taxa, não o mesmo número na mesma escala."
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
