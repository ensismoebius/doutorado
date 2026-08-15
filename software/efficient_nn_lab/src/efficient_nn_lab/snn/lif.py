"""Leaky Integrate-and-Fire neuron (ESPECIFICACAO_DLVL.md #18).

tau dV/dt = -(V - V_rest) + R I(t)

Discretized with a single explicit Euler step per sample, which is enough
to show the qualitative behaviour (leak, integration, threshold, reset)
without needing an ODE solver. R and tau=R*C default to the same values
used in the thesis' worked LIF example (R=5, C=1) so the numbers here are
consistent with the lecture slides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LIFParams:
    tau: float = 5.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_th: float = 1.0
    r: float = 5.0
    dt: float = 1.0


@dataclass
class LIFTrace:
    current: np.ndarray
    membrane: np.ndarray  # membrane potential *after* each step
    spikes: np.ndarray  # 0/1 per step


def simulate_lif(current: np.ndarray, params: LIFParams = LIFParams()) -> LIFTrace:
    """Integrate the LIF equation over a given input-current trace.

    A spike is emitted and V reset to ``v_reset`` the same step it crosses
    ``v_th`` — the membrane potential array therefore never exceeds the
    threshold, matching the sawtooth shape shown in
    ESPECIFICACAO_DLVL.md #18.
    """
    n = len(current)
    v = np.empty(n, dtype=float)
    spikes = np.zeros(n, dtype=float)
    v_prev = params.v_rest
    for t in range(n):
        dv = (-(v_prev - params.v_rest) + params.r * current[t]) * (params.dt / params.tau)
        v_t = v_prev + dv
        if v_t >= params.v_th:
            spikes[t] = 1.0
            v_t = params.v_reset
        v[t] = v_t
        v_prev = v_t
    return LIFTrace(current=np.asarray(current, dtype=float), membrane=v, spikes=spikes)


def constant_current(amplitude: float, n_steps: int, onset: int = 0) -> np.ndarray:
    """A current trace that is zero before ``onset`` and ``amplitude`` after."""
    trace = np.zeros(n_steps, dtype=float)
    trace[onset:] = amplitude
    return trace
