"""Demonstração — regularização de taxa de disparo (software/nn's
.wiki/Concepts/Spike-Rate-Regularization.md).

Both failure modes the regularizer exists for, side by side: a nearly dead
neuron (rate below r_min) and a bursting one (above r_max). An arrow at each
trace's tip is the regularizer's push, -dL_reg/d(rate), in rate units; the
sweep shows it shrinking to zero as each neuron reaches the band edge, which
is the linear restoring force of the formula made visible.

lambda / r_min / r_max default to what the real snn-ae Phase 00 profiles use
(0.5 / 0.10 / 0.80); the starting rates and the update rule (the push applied
straight to the rate) are illustrative stand-ins, said so on screen.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, slider, transition
from efficient_nn_lab.snn.rate_reg import DEFAULT_MAX_RATE, rate_reg_loss, rate_reg_push

_N_EPOCHS = 20
_DEAD_START = 0.02
_BURST_START = 0.95
_LR = 0.3
_PROFILE_LAMBDA = 0.5
_PROFILE_MIN_RATE = 0.10
_EQUATION = "L_reg = lambda [\\max(0, r_min - \\rho)^2 + \\max(0, \\rho - r_max)^2]"


class FiringRateRegDemo(DemoModule):
    title = "SNN -> Regularização de taxa de disparo"
    slug = "snn.firing_rate_reg"
    description = (
        "Um neurônio quase morto e um em rajada recebem um empurrão de volta para a faixa alvo "
        "de taxa de disparo -- lambda e a faixa vêm dos perfis reais; taxas iniciais e passo "
        "são ilustrativos."
    )

    def __init__(self) -> None:
        self.lambda_reg = _PROFILE_LAMBDA
        self.r_min = _PROFILE_MIN_RATE
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "lambda_reg": slider("lambda (peso da regularização)", 0.05, 1.0, 0.05, self.lambda_reg),
            "r_min": slider("r_min (piso da faixa alvo)", 0.01, 0.2, 0.01, self.r_min),
        }

    def _trace(self, start: float) -> np.ndarray:
        rates = [start]
        for _ in range(_N_EPOCHS):
            rate = rates[-1]
            rates.append(float(np.clip(rate + _LR * self._push(rate), 0.0, 1.0)))
        return np.array(rates)

    def _push(self, rate: float) -> float:
        return rate_reg_push(rate, self.lambda_reg, r_min=self.r_min, r_max=DEFAULT_MAX_RATE)

    def _loss(self, rate: float) -> float:
        return rate_reg_loss(rate, self.lambda_reg, r_min=self.r_min, r_max=DEFAULT_MAX_RATE)

    def _build_frames(self) -> list[Frame]:
        dead, burst = self._trace(_DEAD_START), self._trace(_BURST_START)
        epochs = np.arange(_N_EPOCHS + 1)
        lam, r_min, r_max = self.lambda_reg, self.r_min, DEFAULT_MAX_RATE

        def values(k: int, push_reveal: float) -> dict[str, object]:
            return {
                "kind": "firing_rate_reg",
                "epochs": epochs[: k + 1],
                "rate_dead": dead[: k + 1],
                "rate_burst": burst[: k + 1],
                "push_dead": self._push(float(dead[k])),
                "push_burst": self._push(float(burst[k])),
                "loss_dead": self._loss(float(dead[k])),
                "loss_burst": self._loss(float(burst[k])),
                "push_reveal": push_reveal,
                "r_min": r_min,
                "r_max": r_max,
                "n_total": _N_EPOCHS,
                "dead_top": max(0.15, r_min * 1.6),
                "burst_bottom": 0.7,
            }

        if _DEAD_START < r_min:
            dead_line = (
                f"Em cima, taxa {_DEAD_START:.2f}: abaixo do piso r_min = {r_min:.2f}, o neurônio quase "
                "nunca cruza o limiar — e longe do limiar o gradiente substituto é ≈ 0, então ele não "
                "aprende a disparar de novo (como uma ReLU 'morta', só que pior)."
            )
        else:
            dead_line = (
                f"Em cima, taxa {_DEAD_START:.2f}: com r_min = {r_min:.2f} ela já está DENTRO da faixa — "
                "para o regularizador este neurônio não está morto, e nada o empurra. É a faixa que "
                "define o que conta como 'morto'."
            )
        problem = Frame(
            "Dois neurônios doentes",
            values(0, push_reveal=0.0),
            f"{dead_line} Embaixo, taxa {_BURST_START:.2f}: dispara em quase todo quadro, vira uma "
            "constante e não carrega informação nenhuma.",
            _EQUATION,
        )
        push_dead, push_burst = self._push(_DEAD_START), self._push(_BURST_START)
        push = Frame(
            "O empurrão: proporcional à distância até a faixa",
            values(0, push_reveal=1.0),
            "L_reg pune só o quanto a taxa MÉDIA sai da faixa [r_min, r_max], ao quadrado. Seu gradiente "
            f"(setas) é 2·lambda·(distância): com lambda = {lam:g}, {push_dead:+.3f} no de cima e "
            f"{push_burst:+.3f} no de baixo. Ele não depende de onde cai disparo nenhum — por isso "
            "ainda existe quando a perda de reconstrução já não dá gradiente.",
            _EQUATION,
        )
        arrived = Frame(
            "Na borda da faixa, o empurrão some",
            values(_N_EPOCHS, push_reveal=1.0),
            "A cada época a seta encolhe: a força é linear na distância, então cada um se aproxima da "
            f"borda (taxas {dead[-1]:.3f} e {burst[-1]:.3f}) e para nela. Dentro da faixa o gradiente é "
            "exatamente zero: o regularizador só traz o neurônio de volta à faixa — onde ele fica lá "
            "dentro é a perda de reconstrução que decide.",
            _EQUATION,
        )
        caveat = Frame(
            "O que este empurrão não resolve",
            values(_N_EPOCHS, push_reveal=1.0),
            f"lambda = {_PROFILE_LAMBDA:g}, r_min = {_PROFILE_MIN_RATE:.2f} e r_max = {r_max:.2f} são os dos "
            "perfis snn-ae reais do software/nn; taxas iniciais e passo aqui são ilustrativos. No "
            "software/nn o termo age nas camadas Lif do encoder — por isso não resgata sozinho uma "
            "saída de decoder que nunca dispara (demo Perda incompatível com a codificação).",
            _EQUATION,
        )

        frames = [problem]
        frames += transition(problem, push, steps=6)
        frames.append(push)
        for k in range(1, _N_EPOCHS):
            frames.append(Frame(arrived.label, values(k, 1.0), arrived.explanation, _EQUATION, is_checkpoint=False))
        frames += [arrived, caveat]
        return frames
