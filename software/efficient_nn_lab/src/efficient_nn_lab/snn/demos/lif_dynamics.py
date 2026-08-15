"""Demonstração SNN 2 + 3 — Neurônio LIF e integração temporal.

(ESPECIFICACAO_DLVL.md #18, #19, and the mandatory guided sequence #31:
input -> integration -> threshold -> spike -> reset -> repeat.)

Every time-step is a frame, so the membrane trace sweeps smoothly and
continuously — nothing about a leaking, integrating potential should ever
jump. Checkpoints mark only the phase changes that matter didactically
(current onset, each spike+reset, the end), so "Anterior"/"Próximo"
moves between those moments while playback still glides through every
sample in between.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif

_N_STEPS = 60
_ONSET = 5


class LIFDynamicsDemo(DemoModule):
    title = "SNN -> LIF"
    description = "O potencial de membrana integra a corrente de entrada, vaza com o tempo, e dispara ao cruzar o limiar."

    def __init__(self) -> None:
        self.tau = 5.0
        self.r = 5.0
        self.v_th = 1.0
        self.amplitude = 0.30
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "tau": {"label": "tau (constante de tempo)", "min": 1.0, "max": 15.0, "step": 0.5, "value": self.tau},
            "r": {"label": "R (resistência)", "min": 1.0, "max": 10.0, "step": 0.5, "value": self.r},
            "v_th": {"label": "V_th (limiar)", "min": 0.3, "max": 3.0, "step": 0.1, "value": self.v_th},
            "amplitude": {"label": "Amplitude de I(t)", "min": 0.05, "max": 1.0, "step": 0.05, "value": self.amplitude},
        }

    def _build_frames(self) -> list[Frame]:
        params = LIFParams(tau=self.tau, r=self.r, v_th=self.v_th)
        current = constant_current(self.amplitude, _N_STEPS, onset=_ONSET)
        trace = simulate_lif(current, params)

        frames = []
        for t in range(_N_STEPS):
            is_checkpoint = t in (0, _N_STEPS - 1)
            if t < _ONSET:
                phase = "repouso"
                explanation = "Sem corrente de entrada: o potencial permanece em V_rest."
            elif trace.spikes[t] == 1.0:
                phase = "spike + reset"
                explanation = f"V atingiu o limiar V_th={self.v_th:.2f} -> dispara um spike e reinicia em V_reset."
                is_checkpoint = True
            elif t == _ONSET:
                phase = "integração"
                explanation = "A corrente de entrada liga: o potencial começa a subir em direção ao limiar."
                is_checkpoint = True
            elif trace.membrane[t] > (trace.membrane[t - 1] if t > 0 else 0.0):
                phase = "integração"
                explanation = "A corrente de entrada acumula: o potencial sobe em direção ao limiar."
            else:
                phase = "vazamento"
                explanation = "Sem corrente suficiente para compensar o vazamento: o potencial decai."

            frames.append(
                Frame(
                    label=f"t = {t} ({phase})",
                    values={
                        "kind": "lif_trace",
                        "current": trace.current[: t + 1],
                        "membrane": trace.membrane[: t + 1],
                        "spikes": trace.spikes[: t + 1],
                        "v_th": self.v_th,
                        "phase": phase,
                    },
                    explanation=explanation,
                    equation="tau dV/dt = -(V - V_rest) + R . I(t); dispara e reinicia V se V >= V_th",
                    is_checkpoint=is_checkpoint,
                )
            )
        return frames
