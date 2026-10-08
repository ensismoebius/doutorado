"""Demonstração — regularização de taxa de disparo (software/nn's
.wiki/Concepts/Spike-Rate-Regularization.md).

What the regularizer actually sees is ONE number per layer: the mean firing
rate over every unit, frame and sample of that layer's spike tensor
(`SpikeCountLoss`: ``spike_sum / input.size()``; the classifier's
`add_firing_rate_grad`: ``spikes.sum() / n``). Its gradient with respect to
that mean is the same scalar for every unit. So the demo is built around
layers, not neurons:

* two sick layers, side by side -- a nearly dead one (mean below r_min) and
  a bursting one (mean above r_max). The arrow at each trace's tip is the
  push on the layer mean, -dL_reg/d(mean), in rate units; the sweep shows it
  shrinking to zero at the band edge, the linear restoring force of the
  formula made visible;
* then ONE mixed layer holding a dead unit and a bursting unit. Their mean
  lands inside the band, so L_reg = 0 and neither unit is pushed at all --
  the blind spot of a mean-rate penalty, which a per-neuron picture of the
  regularizer would hide.

lambda / r_min / r_max default to what the real snn-ae Phase 00 profiles use
(0.5 / 0.10 / 0.80); the starting rates and the update rule (the push applied
straight to the mean rate) are illustrative stand-ins, said so on screen.
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
_EQUATION = "L_reg = lambda [\\max(0, r_min - \\rho)^2 + \\max(0, \\rho - r_max)^2],\\ \\rho = \\text{média da camada}"


class FiringRateRegDemo(DemoModule):
    title = "SNN -> Regularização de taxa de disparo"
    slug = "snn.firing_rate_reg"
    description = (
        "A penalidade olha a taxa MÉDIA de cada camada: empurra de volta à faixa alvo uma camada "
        "quase morta e uma em rajada -- mas não enxerga uma camada que mistura as duas coisas. "
        "lambda e a faixa vêm dos perfis reais; taxas iniciais e passo são ilustrativos."
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
        """Mean rate of one layer per epoch, moved by the push on that mean."""
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
        # One layer, two units: one nearly silent, one bursting. The
        # regularizer only ever sees their mean.
        mixed_mean = (_DEAD_START + _BURST_START) / 2.0
        mixed_push = self._push(mixed_mean)
        mixed_loss = self._loss(mixed_mean)

        def values(k: int, push_reveal: float, mixed_reveal: float = 0.0) -> dict[str, object]:
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
                "mixed_reveal": mixed_reveal,
                "mixed_dead": _DEAD_START,
                "mixed_burst": _BURST_START,
                "mixed_mean": mixed_mean,
                "mixed_push": mixed_push,
                "r_min": r_min,
                "r_max": r_max,
                "n_total": _N_EPOCHS,
                "dead_top": max(0.15, r_min * 1.6),
                "burst_bottom": 0.7,
            }

        if _DEAD_START < r_min:
            dead_line = (
                f"Em cima, uma camada quase morta: média {_DEAD_START:.2f}, abaixo do piso r_min = "
                f"{r_min:.2f}. Suas unidades quase nunca cruzam o limiar — e longe do limiar o "
                "gradiente substituto é ≈ 0, então elas não aprendem a disparar de novo (como uma "
                "ReLU 'morta', só que pior)."
            )
        else:
            dead_line = (
                f"Em cima, média {_DEAD_START:.2f}: com r_min = {r_min:.2f} ela já está DENTRO da faixa "
                "— para o regularizador esta camada não está morta, e nada a empurra. É a faixa que "
                "define o que conta como 'morto'."
            )
        problem = Frame(
            "Duas camadas doentes",
            values(0, push_reveal=0.0),
            "O regularizador não olha neurônio por neurônio: ele mede UMA taxa por camada, a média "
            "sobre todas as unidades, quadros e amostras do lote (no software/nn, soma dos spikes ÷ "
            f"número de elementos). {dead_line} Embaixo, uma camada em rajada: média "
            f"{_BURST_START:.2f}, dispara em quase todo quadro, vira uma constante e não carrega "
            "informação nenhuma.",
            _EQUATION,
        )
        push_dead, push_burst = self._push(_DEAD_START), self._push(_BURST_START)
        push = Frame(
            "O empurrão: proporcional à distância até a faixa",
            values(0, push_reveal=1.0),
            "L_reg pune só o quanto a média da camada sai da faixa [r_min, r_max], ao quadrado. Seu "
            f"gradiente em relação a essa média (setas) é 2·lambda·(distância): com lambda = {lam:g}, "
            f"{push_dead:+.3f} na de cima e {push_burst:+.3f} na de baixo. Como a média soma todas as "
            "unidades com o mesmo peso, o MESMO empurrão chega a cada unidade da camada. Ele não "
            "depende de onde cai disparo nenhum — por isso ainda existe quando a perda de "
            "reconstrução já não dá gradiente.",
            _EQUATION,
        )
        arrived = Frame(
            "Na borda da faixa, o empurrão some",
            values(_N_EPOCHS, push_reveal=1.0),
            "A cada época a seta encolhe: a força é linear na distância, então cada média se aproxima "
            f"da borda (médias {dead[-1]:.3f} e {burst[-1]:.3f}) e para nela. Dentro da faixa o "
            "gradiente é exatamente zero: o regularizador só traz a média de volta à faixa — onde ela "
            "fica lá dentro é a perda de reconstrução que decide.",
            _EQUATION,
        )
        mixed = Frame(
            "Camada mista: a média esconde os dois doentes",
            values(_N_EPOCHS, push_reveal=1.0, mixed_reveal=1.0),
            f"Agora UMA camada com uma unidade quase morta ({_DEAD_START:.2f}) e outra em rajada "
            f"({_BURST_START:.2f}), as linhas tracejadas. A média é ({_DEAD_START:.2f} + "
            f"{_BURST_START:.2f})/2 = {mixed_mean:.3f}, dentro da faixa: $L_reg = {mixed_loss:g}$ e o "
            f"empurrão é {mixed_push:+.3f}. Nenhuma das duas é corrigida, e nada avisa — falha "
            "silenciosa. Uma penalidade sobre a média enxerga a camada, não a unidade; para pegar "
            "unidades isoladas é preciso penalizar a taxa de cada unidade.",
            _EQUATION,
        )
        caveat = Frame(
            "O que este empurrão não resolve",
            values(_N_EPOCHS, push_reveal=1.0, mixed_reveal=1.0),
            f"lambda = {_PROFILE_LAMBDA:g}, r_min = {_PROFILE_MIN_RATE:.2f} e r_max = {r_max:.2f} são os "
            "dos perfis snn-ae reais do software/nn; taxas iniciais e passo aqui são ilustrativos (o "
            "empurrão é aplicado direto na média). No modelo real ele chega à saída de spikes de "
            "cada unidade e ainda precisa atravessar o gradiente substituto até os pesos: uma unidade "
            "muito abaixo do limiar recebe pouco ou nada dele. No software/nn o termo age nas "
            "camadas Lif do encoder — por isso não resgata sozinho uma saída de decoder que nunca "
            "dispara (demo Perda incompatível com a codificação).",
            _EQUATION,
        )

        frames = [problem]
        frames += transition(problem, push, steps=6)
        frames.append(push)
        for k in range(1, _N_EPOCHS):
            frames.append(Frame(arrived.label, values(k, 1.0), arrived.explanation, _EQUATION, is_checkpoint=False))
        frames.append(arrived)
        frames += transition(arrived, mixed, steps=6)
        frames += [mixed, caveat]
        return frames
