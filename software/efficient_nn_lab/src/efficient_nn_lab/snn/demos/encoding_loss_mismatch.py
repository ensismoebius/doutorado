"""Demonstração — a perda tem que medir o lugar onde a codificação guarda o
valor (software/nn's .wiki/Concepts/Spike-Encoding.md, "Critical
Invariant: Encoding Must Match Loss" + "The No-Spike Problem").

Two different failures, kept apart on purpose -- the earlier version of
this demo labelled the second one as the first, which the wiki does not
support:

* Act 1, the MISMATCH: a latency-coded unit trained with SpikeCountLoss.
  Prediction and target each fire once, so the count loss is 0 from the
  first epoch and the spike never moves off the wrong frame.
* Act 2, the NO-SPIKE DEADLOCK, with the CORRECT pair (latency +
  SpikeTimeLoss): a unit that never fires is scored at t = T, and
  `if (t < T)` in the backward pass writes no gradient for it at all.

The spike frame is a rounded stand-in for a gradient-trained latent
(p <- p + lr (target - p)), so the plotted frames stay integers, as real
first-spike frames are.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.snn.encoding import spike_time_grad_is_live

_TIME_STEPS = 16
_TARGET_FRAME = 4
_START_FRAME = 12
_N_EPOCHS = 12
_LR = 0.35
_EQUATION = "L_{tempo} = (t_{prev} - t_{alvo})^2;  L_{contagem} = (n_{prev} - n_{alvo})^2"


def _spike_time_run(never_fires: bool) -> np.ndarray:
    """First-spike frame per epoch under SpikeTimeLoss.

    The update only happens when the backward pass would write a gradient
    -- `spike_time_grad_is_live`, the same `if (t < T)` guard as
    SpikeTimeLossImpl::backward -- so the silent unit's flat line comes out
    of the guard itself instead of being drawn by hand.
    """
    latent = float(_START_FRAME)
    frames = []
    for _ in range(_N_EPOCHS):
        frame = _TIME_STEPS if never_fires else int(np.floor(latent + 0.5))
        frames.append(frame)
        if spike_time_grad_is_live(frame, _TIME_STEPS):
            latent += _LR * (_TARGET_FRAME - latent)
    return np.array(frames, dtype=float)


def _spike_count_run() -> tuple[np.ndarray, np.ndarray]:
    """Same unit under SpikeCountLoss: (frames, losses) per epoch.

    A latency code fires exactly once, in the target and in the prediction,
    so the count error -- and with it the gradient -- is 0 at every epoch,
    and nothing ever moves the spike off its starting frame.
    """
    predicted_count = target_count = 1
    losses = np.full(_N_EPOCHS, float((predicted_count - target_count) ** 2))
    return np.full(_N_EPOCHS, float(_START_FRAME)), losses


class EncodingLossMismatchDemo(DemoModule):
    title = "SNN -> Perda incompatível com a codificação"
    slug = "snn.encoding_loss_mismatch"
    description = (
        "Numa SNN (Spiking Neural Network, rede neural de pulso), a perda tem que medir o lugar "
        "onde a codificação guarda o valor. Com a perda errada, o treino reporta 'perfeito' sem "
        "corrigir nada; com a certa, uma unidade que nunca dispara também trava. Duas falhas "
        "silenciosas."
    )

    def _build_frames(self) -> list[Frame]:
        epochs = np.arange(_N_EPOCHS)
        moved = _spike_time_run(never_fires=False)
        moved_loss = (moved - _TARGET_FRAME) ** 2
        stuck_wrong, count_loss = _spike_count_run()
        never = _spike_time_run(never_fires=True)
        never_loss = (never - _TARGET_FRAME) ** 2

        act1 = {
            "act_title": "Ato 1 — mesma unidade de latência, duas perdas diferentes",
            "frame_a": moved, "loss_a": moved_loss, "label_a": "SpikeTimeLoss",
            "frame_b": stuck_wrong, "loss_b": count_loss, "label_b": "SpikeCountLoss",
            "never_reveal": 0.0, "loss_max": float(moved_loss.max()),
        }
        act2 = {
            "act_title": "Ato 2 — par certo (latência + SpikeTimeLoss), uma unidade muda",
            "frame_a": moved, "loss_a": moved_loss, "label_a": "unidade A",
            "frame_b": never, "loss_b": never_loss, "label_b": "unidade B",
            "never_reveal": 1.0, "loss_max": float(never_loss.max()),
        }
        n0 = int(moved_loss[0])
        stops_act1 = {
            0: (
                "Cada codificação guarda o valor num lugar diferente",
                "Latência guarda o valor no MOMENTO do disparo (um disparo por unidade); Poisson, na "
                "CONTAGEM de disparos. A perda precisa medir esse mesmo lugar. Teste: o alvo é disparar "
                f"no quadro {_TARGET_FRAME} (de T = {_TIME_STEPS}) e a unidade começa no quadro "
                f"{_START_FRAME}. Duas execuções idênticas; só a perda muda.",
            ),
            _N_EPOCHS - 1: (
                "Perda errada: SpikeCountLoss diz 'perfeito' com o disparo errado",
                f"SpikeTimeLoss mede o tempo: ({_START_FRAME} − {_TARGET_FRAME})² = {n0}, e o gradiente "
                f"puxa o disparo até o quadro {_TARGET_FRAME} (verde). SpikeCountLoss só conta: 1 disparo "
                "previsto, 1 no alvo, perda 0 desde a 1ª época — gradiente zero, e o disparo fica no "
                f"quadro {_START_FRAME} para sempre (vermelho). A perda reporta 'perfeito' com erro real "
                f"de {n0}: falha silenciosa.",
            ),
        }
        stops_act2 = {
            0: (
                "Mesmo com a perda certa: a unidade que nunca dispara",
                "Segunda armadilha, agora com o par CERTO (latência + SpikeTimeLoss). A unidade B nunca "
                "cruza o limiar: não há disparo. Por convenção, SpikeTimeLoss conta 'nunca disparou' "
                f"como disparo em t = T = {_TIME_STEPS} — assim a perda fica finita: "
                f"({_TIME_STEPS} − {_TARGET_FRAME})² = {int(never_loss[0])}.",
            ),
            _N_EPOCHS - 1: (
                "Sem disparo, sem gradiente: o deadlock",
                "O backward só escreve gradiente na linha do disparo previsto: if (t < T). Sem disparo, "
                "t = T, a condição falha e NADA é escrito — a unidade nunca recebe o sinal 'dispare mais "
                f"cedo'. O treino termina, reporta {int(never_loss[0])} e não mudou nada: silêncio → "
                "gradiente zero → silêncio para sempre.",
            ),
        }
        loud = (
            "No software/nn: de silencioso a barulhento",
            "No software/nn as duas falhas viraram erros: ThesisConfig::validate() recusa o par "
            "codificação/perda errado antes de treinar, e assert_gradients_were_live() aborta o treino "
            "se TODOS os lotes tiveram gradiente zero. Remédios para o deadlock, em ordem: taxa de "
            "aprendizado maior, V_th menor, mais time_steps, tdBN antes da camada.",
        )

        frames: list[Frame] = []
        for act, stops in ((act1, stops_act1), (act2, stops_act2)):
            for k in range(_N_EPOCHS):
                label, explanation = stops[min(s for s in stops if s >= k)]
                frames.append(
                    Frame(
                        label=label,
                        values=self._values(act, epochs, k + 1),
                        explanation=explanation,
                        equation=_EQUATION,
                        is_checkpoint=k in stops,
                    )
                )
        frames.append(Frame(loud[0], self._values(act2, epochs, _N_EPOCHS), loud[1], _EQUATION))
        return frames

    @staticmethod
    def _values(act: dict[str, object], epochs: np.ndarray, cut: int) -> dict[str, object]:
        series = ("frame_a", "loss_a", "frame_b", "loss_b")
        return {
            "kind": "encoding_loss_mismatch",
            **{k: v for k, v in act.items() if k not in series},
            **{k: np.asarray(act[k])[:cut] for k in series},
            "iterations": epochs[:cut],
            "n_total": _N_EPOCHS - 1,
            "time_steps": _TIME_STEPS,
            "target_frame": _TARGET_FRAME,
        }
