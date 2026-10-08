"""Demonstração SNN 2 + 3 — Neurônio LIF e integração temporal.

(ESPECIFICACAO_DLVL.md #18, #19, and the mandatory guided sequence #31:
input -> integration -> threshold -> spike -> reset -> repeat.)

Every time-step is a frame, so the membrane trace sweeps smoothly and
continuously — nothing about a leaking, integrating potential should ever
jump. Checkpoints mark only the phase changes that matter didactically
(current onset, each spike+reset, current offset, the end), so
"Anterior"/"Próximo" moves between those moments while playback still
glides through every sample in between.

The input is a current PULSE, not a step: while a constant current is on,
V only rises (after every reset it starts below the equilibrium R·I), so
the leak never shows on its own. Once the current switches off, the leak
is the only thing acting and V visibly decays back to V_rest -- the "Leaky"
in LIF. With R·I below V_th the text says why the neuron never fires: the
leak grows with V and balances the input at V = R·I.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, slider
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif

_N_STEPS = 60
_ONSET = 5
#: Switch-off frame. With the default tau = R = 5, V_th = 1, I = 0.3 the
#: neuron fires every 5 steps (t = 9, 14, ..., 34); switching off at 39
#: catches V high (~0.89), so the decay is large and easy to see. At 40 a
#: spike at t = 39 would have reset V to 0 first, leaving nothing to leak.
_OFFSET = 39


class LIFDynamicsDemo(DemoModule):
    title = "SNN -> LIF"
    slug = "snn.lif"
    description = (
        "O potencial de membrana integra a corrente de entrada, dispara ao cruzar o limiar e "
        "reinicia; quando a corrente desliga, o vazamento traz o potencial de volta ao repouso."
    )

    def __init__(self) -> None:
        self.tau = 5.0
        self.r = 5.0
        self.v_th = 1.0
        self.amplitude = 0.30
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "tau": slider("tau (constante de tempo)", 1.0, 15.0, 0.5, self.tau),
            "r": slider("R (resistência)", 1.0, 10.0, 0.5, self.r),
            "v_th": slider("V_th (limiar)", 0.3, 3.0, 0.1, self.v_th),
            "amplitude": slider("Amplitude de I(t)", 0.05, 1.0, 0.05, self.amplitude),
        }

    def _build_frames(self) -> list[Frame]:
        params = LIFParams(tau=self.tau, r=self.r, v_th=self.v_th)
        current = constant_current(self.amplitude, _N_STEPS, onset=_ONSET, offset=_OFFSET)
        trace = simulate_lif(current, params)
        equilibrium = self.r * self.amplitude  # where the leak balances the input: -(V - 0) + R·I = 0
        reaches_threshold = equilibrium >= self.v_th

        # Every explanation is kept to ONE line: the window reserves the
        # height of this demo's tallest checkpoint text, and LIF already
        # gives up height to its 4 sliders (see
        # test_lif_canvas_not_shrunk_by_short_explanations).
        if reaches_threshold:
            onset = "A corrente de entrada liga: o potencial começa a subir em direção ao limiar."
            rising = "A corrente de entrada acumula: o potencial sobe em direção ao limiar."
        else:
            # Shown from the onset checkpoint on, not only between
            # checkpoints: the height reservation measures checkpoint
            # texts only, so a text no checkpoint shows is never measured.
            onset = rising = (
                "O vazamento cresce com V e equilibra a entrada em "
                f"$R · I = {equilibrium:.2f} < V_th = {self.v_th:.2f}$: nunca dispara."
            )

        frames = []
        for t in range(_N_STEPS):
            is_checkpoint = t in (0, _N_STEPS - 1)
            if t < _ONSET:
                phase = "repouso"
                explanation = "Sem corrente de entrada: o potencial permanece em $V_rest$."
            elif trace.spikes[t] == 1.0:
                phase = "spike + reset"
                explanation = f"V atingiu o limiar $V_th = {self.v_th:.2f}$ -> dispara um spike e reinicia em $V_reset$."
                is_checkpoint = True
            elif t == _ONSET:
                phase = "integração"
                explanation = onset
                is_checkpoint = True
            elif t < _OFFSET:
                phase = "integração"
                explanation = rising
            else:
                phase = "vazamento"
                explanation = (
                    "Corrente desligada: só o vazamento age, e V decai exponencialmente de volta a $V_rest$."
                )
                is_checkpoint = is_checkpoint or t == _OFFSET

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
