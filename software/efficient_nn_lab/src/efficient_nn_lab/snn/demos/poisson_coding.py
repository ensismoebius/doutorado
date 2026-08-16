"""Demonstração SNN — Codificação Poisson (rate coding).

Single question answered: how does an SNN turn a real-valued input into
spikes when the value isn't just "above or below a level," but a rate? This
sits next to spike_generation.py's `direct_threshold_spikes` on purpose —
same synthetic signal, same time axis, so the two demos are directly
comparable: one produces a spike deterministically at every rising edge,
the other produces spikes probabilistically, more often where the signal
is more intense, but never guaranteed at any single step.

The draw is seeded (ESPECIFICACAO_DLVL.md #35), so "Anterior"/"Próximo"/
reset always reproduce the exact same spike train — the *mechanism* is
probabilistic, the *demo* is fully deterministic to replay.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.snn.encoding import poisson_spikes, spike_probability, synthetic_signal

_N_STEPS = 60


class PoissonCodingDemo(DemoModule):
    title = "SNN -> Codificação Poisson"
    slug = "snn.poisson"
    description = (
        "O mesmo sinal contínuo, mas cada passo de tempo só dispara com uma probabilidade "
        "proporcional à intensidade — não mais um cruzamento de nível garantido."
    )

    def __init__(self) -> None:
        self.max_rate = 0.9
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "max_rate": {
                "label": "Taxa máxima de disparo",
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "value": self.max_rate,
            }
        }

    def _build_frames(self) -> list[Frame]:
        signal = synthetic_signal(_N_STEPS)
        prob = spike_probability(signal, self.max_rate)
        spikes = poisson_spikes(signal, self.max_rate)
        spike_times = {i for i, s in enumerate(spikes) if s > 0}

        frames: list[Frame] = []
        for t in range(_N_STEPS):
            n_spikes_so_far = int(spikes[: t + 1].sum())
            p_now = float(prob[t])
            explanation = (
                f"P(spike) agora = {p_now:.2f} (intensidade x taxa máxima {self.max_rate:.2f}). "
                f"{n_spikes_so_far} spike(s) sorteado(s) até agora — mesma intensidade não "
                "garante o mesmo resultado a cada repetição, só a mesma tendência."
            )
            is_checkpoint = t == 0 or t == _N_STEPS - 1 or t in spike_times
            frames.append(
                Frame(
                    label=f"t = {t}/{_N_STEPS - 1}",
                    values={
                        "kind": "poisson_spikes",
                        "signal": signal[: t + 1],
                        "prob": prob[: t + 1],
                        "spikes": spikes[: t + 1],
                        "n_total": _N_STEPS,
                        "max_rate": self.max_rate,
                    },
                    explanation=explanation,
                    equation="P(spike no passo t) = intensidade(t) . taxa_máxima",
                    is_checkpoint=is_checkpoint,
                )
            )
        return frames
