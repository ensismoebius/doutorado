"""Demonstração — o plano paraconsistente (software/nn's thesis-novel
paraconsistent feature engineering, .wiki/Core/Paraconsistent.md).

Single question answered: given how compact each class is (alpha) and how
much classes overlap (beta), where does that land on the paraconsistent
plane, and how far is it from "Verdade" (perfectly separable)?

The two coefficients are applied one at a time so each one's geometry is
seen on its own: from alpha = beta = 0 (Indefinição), alpha alone moves the
point along the Indefinição -> Verdade diagonal (G1 = alpha, G2 = alpha - 1);
beta then moves it along the perpendicular (-1, +1) direction, taking the
same amount from certainty that it adds to contradiction.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence, slider
from efficient_nn_lab.paraconsistent.metrics import ParaconsistentPoint, score_point

_TWEEN_STEPS = 12


def _contradiction_reading(g2: float) -> str:
    if abs(g2) < 0.1:
        return "G2 perto de 0: compacidade e sobreposição não se contradizem."
    if g2 > 0:
        return "G2 > 0: classes compactas, mas sobrepostas — evidências em conflito (lado da Ambiguidade)."
    return "G2 < 0: classes espalhadas e pouco sobrepostas — pouca evidência de qualquer coisa (lado da Indefinição)."


class ParaconsistentPlaneDemo(DemoModule):
    title = "Paraconsistente -> Plano paraconsistente"
    slug = "paraconsistent.plane"
    description = (
        "Alpha mede o quão compacta cada classe é; beta mede o quanto as classes se "
        "sobrepõem. Juntos, definem um ponto no plano paraconsistente, com quatro "
        "cantos: Verdade, Falsidade, Ambiguidade e Indefinição."
    )

    def __init__(self) -> None:
        self.alpha = 0.92
        self.beta = 0.075
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "alpha": slider("alpha (compacidade intraclasse)", 0.0, 1.0, 0.01, self.alpha),
            "beta": slider("beta (sobreposição interclasse)", 0.0, 1.0, 0.01, self.beta),
        }

    def _build_frames(self) -> list[Frame]:
        a, b = float(self.alpha), float(self.beta)
        origin, only_alpha, point = score_point(0.0, 0.0), score_point(a, 0.0), score_point(a, b)

        def frame(p: ParaconsistentPoint, label: str, explanation: str, equation: str, distance: float) -> Frame:
            return Frame(
                label,
                {
                    "kind": "paraconsistent_plane",
                    "g1": p.g1, "g2": p.g2, "d_truth": p.d_truth, "d_penalized": p.d_penalized,
                    "distance_reveal": distance,
                },
                explanation,
                equation,
            )

        alpha_eq = "alpha = 1 - \\max_{n} \\overline{svC_n}"
        checkpoints = [
            frame(
                origin,
                "Antes de treinar: estas features separam as classes?",
                "A tese responde isso ANTES de treinar qualquer classificador, com dois números em "
                "[0, 1]: α (o quanto cada classe é compacta) e β (o quanto as classes invadem umas às "
                "outras). Eles viram um ponto neste plano, cujos cantos são os extremos: Verdade "
                "(compactas e separadas), Falsidade (tudo misturado), Ambiguidade (compactas, mas no "
                "mesmo lugar), Indefinição (sem estrutura). O ponto parte de α = β = 0.",
                alpha_eq,
                0.0,
            ),
            frame(
                only_alpha,
                f"α = {a:.2f}: o quanto cada classe é compacta",
                "Em cada classe e cada dimensão mede-se a faixa ocupada (máximo − mínimo) e tira-se a "
                f"média; α = 1 − a maior dessas médias. α = {a:.2f}: até a classe mais espalhada ocupa, "
                f"em média, só {100 * (1 - a):.0f}% da faixa [0, 1]. Sozinho, α leva o ponto pela "
                f"diagonal Indefinição → Verdade: G1 = α = {only_alpha.g1:.2f}, G2 = α − 1 = "
                f"{only_alpha.g2:.2f}.",
                alpha_eq,
                0.0,
            ),
            frame(
                point,
                f"β = {b:.3f}: o quanto as classes se invadem",
                f"β é a fração dos valores de uma classe que caem dentro da faixa de outra: {100 * b:.1f}%. "
                "Cada unidade de β tira o mesmo da certeza que soma à contradição — o ponto anda na "
                f"perpendicular: G1 = α − β = {point.g1:.3f} (certeza), G2 = α + β − 1 = {point.g2:.3f} "
                f"(contradição). {_contradiction_reading(point.g2)}",
                "G1 = alpha - beta;  G2 = alpha + beta - 1",
                0.0,
            ),
            frame(
                point,
                "D_truth: a distância até Verdade",
                "Quanto mais perto de Verdade, melhor o conjunto de features — medido sem treinar nada: "
                f"D_truth = √((G1 − 1)² + G2²) = √(({point.g1:.3f} − 1)² + ({point.g2:.3f})²), ou seja "
                f"$D_truth = {point.d_truth:.4f}$. Ranquear extratores por esse número parece "
                "suficiente; a próxima demo mostra o caso em que ele engana.",
                "D_truth = \\sqrt{(G1 - 1)^2 + G2^2}",
                1.0,
            ),
        ]
        return build_sequence(checkpoints, steps=_TWEEN_STEPS)
