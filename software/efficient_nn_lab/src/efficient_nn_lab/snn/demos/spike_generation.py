"""Demonstração SNN 1 — Sinal e spikes (ESPECIFICACAO_DLVL.md #17).

Single question answered: what is a spike, at the most basic level? A
continuous signal is revealed left-to-right, one time-step at a time; a
spike appears every time it crosses a fixed level on the way up. No
membrane dynamics yet — that is the next demo (lif_dynamics).

Every one of the 60 time-steps is a frame (so playback sweeps smoothly,
never jumping half the signal at once), but only the start, each spike,
and the end are *checkpoints* — what "Anterior"/"Próximo" jump between,
so manual navigation moves meaningfully instead of one sample at a time.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.snn.encoding import direct_threshold_spikes, synthetic_signal

_N_STEPS = 60


class SpikeGenerationDemo(DemoModule):
    title = "SNN -> Sinal e spikes"
    slug = "snn.spikes"
    description = "Um sinal contínuo cruza um nível; cada cruzamento de subida produz um spike."

    def __init__(self) -> None:
        self.level = 0.4
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "level": {"label": "Nivel de disparo", "min": 0.1, "max": 0.9, "step": 0.05, "value": self.level}
        }

    def _build_frames(self) -> list[Frame]:
        signal = synthetic_signal(_N_STEPS)
        spikes = direct_threshold_spikes(signal, self.level)
        spike_times = {i for i, s in enumerate(spikes) if s > 0}

        frames: list[Frame] = []
        for t in range(_N_STEPS):
            n_spikes_so_far = int(spikes[: t + 1].sum())
            is_checkpoint = t == 0 or t == _N_STEPS - 1 or t in spike_times
            explanation = (
                f"{n_spikes_so_far} spike(s) até agora — um a cada cruzamento de subida do nível {self.level:.2f}."
                if n_spikes_so_far
                else f"Nenhum spike ainda — aguardando o sinal cruzar o nível {self.level:.2f} subindo."
            )
            frames.append(
                Frame(
                    label=f"t = {t}/{_N_STEPS - 1}",
                    values={
                        "kind": "signal_spikes",
                        "signal": signal[: t + 1],
                        "spikes": spikes[: t + 1],
                        "level": self.level,
                        "n_total": _N_STEPS,
                    },
                    explanation=explanation,
                    is_checkpoint=is_checkpoint,
                )
            )
        return frames
