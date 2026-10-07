"""Demonstração — D_truth × D_penalized: por que a distância ingênua até
"Verdade" pode ser enganada por um extrator "morto" (software/nn's
.wiki/Core/Paraconsistent.md -- flagship contribution #1 da tese).

Three extractors, measured one at a time (each fades in while gliding out
of a shared placeholder): a good one (alpha = 0.92, beta = 0.075, the
reference case of software/nn's paraconsistent tests), a weak-but-real one
(alpha = 0.2, beta = 0.7, illustrative) and a dead one (alpha = beta = 1).
The weak one is what makes the exploit visible: the dead extractor's
D_truth = sqrt(2) does NOT beat the good one (0.155), it beats any real
feature set with D_truth > sqrt(2) -- which a two-point demo cannot show.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.paraconsistent.metrics import CONTRADICTION_PENALTY, ParaconsistentPoint, score_point

_GOOD = (0.92, 0.075)
_WEAK = (0.2, 0.7)
_DEAD = (1.0, 1.0)
_TWEEN_STEPS = 12
# Where an extractor sits before it is measured: score_point(0, 0) is the
# Indefinição vertex (g1 = 0, g2 = -1) -- computed, not hard-coded.
_UNMEASURED = score_point(0.0, 0.0)
_TRUTH_EQUATION = "D_truth = \\sqrt{(G1 - 1)^2 + G2^2}"
_PENALIZED_EQUATION = "D_penalized = D_truth + lambda |G2|,  lambda = 2 - \\sqrt{2}"


def _point(prefix: str, p: ParaconsistentPoint, opacity: float) -> dict[str, float]:
    return {
        f"{prefix}g1": p.g1, f"{prefix}g2": p.g2,
        f"{prefix}d_truth": p.d_truth, f"{prefix}d_penalized": p.d_penalized,
        f"{prefix}opacity": opacity,
    }


class ParaconsistentDPenalizedDemo(DemoModule):
    title = "Paraconsistente -> D_truth x D_penalized"
    slug = "paraconsistent.dpenalized"
    description = (
        "Um extrator de features 'morto' (mesma saída sempre, ignorando a classe) cai no canto "
        "Ambiguidade e, por D_truth, passa à frente de extratores fracos porém reais. "
        "D_penalized fecha essa brecha."
    )

    def _build_frames(self) -> list[Frame]:
        good, weak, dead = score_point(*_GOOD), score_point(*_WEAK), score_point(*_DEAD)
        state: dict[str, object] = {
            "kind": "paraconsistent_plane",
            "main_label": "bom",
            **_point("", _UNMEASURED, 0.0),
            **_point("weak_", _UNMEASURED, 0.0),
            **_point("dead_", _UNMEASURED, 0.0),
            "distance_reveal": 1.0,
            "rank_truth": 0.0,
            "rank_penalized": 0.0,
            "vertex_scores": 0.0,
        }

        def checkpoint(label: str, explanation: str, equation: str, **changes: object) -> Frame:
            state.update(changes)
            return Frame(label, dict(state), explanation, equation)

        checkpoints = [
            checkpoint(
                "Ranquear extratores pela distância até Verdade",
                "A tese escolhe o extrator de features pela distância do seu ponto até Verdade — sem "
                "treinar classificador nenhum. Vamos medir três extratores, um de cada vez, e conferir "
                "se o ranking faz sentido. Antes de medir, nenhum tem ponto no plano.",
                _TRUTH_EQUATION,
            ),
            checkpoint(
                f"Um extrator bom: α = {_GOOD[0]}, β = {_GOOD[1]}",
                f"Classes compactas (α = {_GOOD[0]}) e quase sem sobreposição (β = {_GOOD[1]}): o ponto "
                f"cai perto de Verdade, $D_truth = {good.d_truth:.4f}$ — o caso de referência dos testes "
                "do software/nn. É o que todo extrator quer ser.",
                _TRUTH_EQUATION,
                **_point("", good, 1.0),
            ),
            checkpoint(
                f"Um extrator fraco, mas real: α = {_WEAK[0]}, β = {_WEAK[1]}",
                f"Classes espalhadas (α = {_WEAK[0]}) e muito sobrepostas (β = {_WEAK[1]}) — valores "
                "ilustrativos. Ruim, mas carrega ALGUMA informação de classe: as classes não são "
                f"idênticas (β < 1). Fica longe de Verdade: $D_truth = {weak.d_truth:.4f}$.",
                _TRUTH_EQUATION,
                **_point("weak_", weak, 1.0),
            ),
            checkpoint(
                "Um extrator morto: α = β = 1",
                "Um extrator 'morto' emite o MESMO vetor para toda amostra, de qualquer classe. "
                "Espalhamento zero dentro de cada classe: α = 1. Todas as classes no mesmo ponto: β = 1. "
                f"Cai no canto Ambiguidade, com $D_truth = {dead.d_truth:.4f}$ (√2) e zero informação de "
                "classe. Não é hipotético: um SNN-AE treinado com taxa de aprendizado baixa chega "
                "exatamente aqui.",
                _TRUTH_EQUATION,
                **_point("dead_", dead, 1.0),
            ),
            checkpoint(
                "Ranking por D_truth: o morto passa à frente do fraco",
                f"Ordenando por D_truth (embaixo): bom {good.d_truth:.3f} < morto {dead.d_truth:.3f} < "
                f"fraco {weak.d_truth:.3f}. O extrator sem informação nenhuma vence um que tem alguma — "
                "qualquer conjunto real com D_truth acima de √2 perde para ele. Nada avisa: o número é "
                "plausível e o ranking parece normal. Falha silenciosa.",
                _TRUTH_EQUATION,
                rank_truth=1.0,
            ),
            checkpoint(
                "D_penalized: a contradição custa caro",
                "A correção soma uma penalidade pela contradição |G2|, cujos extremos (±1) são "
                "justamente os cantos degenerados Ambiguidade e Indefinição: D_penalized = D_truth + "
                f"λ·|G2|. Agora: bom {good.d_penalized:.3f} < fraco {weak.d_penalized:.3f} < morto "
                f"{dead.d_penalized:.3f}. O bom quase não muda ({good.d_truth:.4f} → "
                f"{good.d_penalized:.4f}): ele quase não tem contradição.",
                _PENALIZED_EQUATION,
                rank_penalized=1.0,
            ),
            checkpoint(
                f"Por que λ = 2 − √2 ≈ {CONTRADICTION_PENALTY:.3f}",
                "Com esse λ os três cantos ruins valem exatamente 2: Ambiguidade e Indefinição, "
                "√2 + (2 − √2)·1 = 2; Falsidade, D_truth = 2 com G2 = 0. Verdade vale 0, e 2 é o pior "
                "valor do plano: nenhum canto degenerado fica 'menos ruim' que outro. E o D_truth "
                "continua inteiro na fórmula porque, sem ele, uma solução degenerada fugiria para "
                "Falsidade.",
                _PENALIZED_EQUATION,
                vertex_scores=1.0,
            ),
        ]
        return build_sequence(checkpoints, steps=_TWEEN_STEPS)
