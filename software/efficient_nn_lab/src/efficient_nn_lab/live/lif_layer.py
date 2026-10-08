"""A vectorized, multi-neuron, trainable LIF layer -- the piece that does
not exist anywhere else in this repo.

`snn/lif.py` simulates exactly one neuron from one scalar current trace,
with no weight matrix and no training; it exists to teach the membrane
equation, not to be a building block. This module is the opposite: many
neurons, a real weight matrix, and a full backward pass, so a small
network of these layers can actually be trained (see `vowel_model.py` and
`train.py`).

Forward recurrence (one layer, every neuron at once), reusing the exact
reset-by-multiplication form already cited in this app's own bibliography
(Zheng et al., 2021, quoted in `snn/demos/tdbn_demo.py`):

    V_t = decay * V_{t-1} * (1 - S_{t-1}) + X_t @ W
    S_t = heaviside(V_t - v_th)

``decay = exp(-dt/tau)`` is the same leak `snn/lif.py` uses, just reused
here per-population instead of per-neuron-scalar. ``(1 - S_{t-1})`` is the
reset: a neuron that just fired starts the next step from zero instead of
from wherever V happened to be.

Backward pass (BPTT): the forward step is a chain of three multiplications
of V_{t-1} -- by `decay`, by `(1 - S_{t-1})`, and implicitly by nothing else
-- plus an addition of the new input. `fast_sigmoid_surrogate`
(`snn/surrogate.py`) stands in for dS/dV, exactly as the existing
surrogate-gradient demo already explains. The ONE deliberate approximation,
named here rather than hidden: the reset gate's own dependence on S_{t-1}
is treated as constant (not differentiated through) when propagating the
gradient backward in time. This is standard surrogate-gradient-SNN practice
(it is what `snn/surrogate.py` already implies by giving the *true*
derivative of a spike as zero everywhere) -- but it is an approximation,
not an identity, and a different choice here would also be defensible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from efficient_nn_lab.snn.surrogate import fast_sigmoid_surrogate, heaviside


@dataclass
class LIFLayerParams:
    tau: float = 5.0
    v_th: float = 1.0
    dt: float = 1.0
    #: Steepness of the surrogate gradient (`snn/surrogate.py`'s own `k`).
    surrogate_k: float = 5.0

    @property
    def decay(self) -> float:
        return math.exp(-self.dt / self.tau)


@dataclass
class LayerCache:
    """Everything `backward_layer` needs that `forward_layer` already
    computed -- kept separate from the (T, n_out) spikes array itself so
    callers that only need the forward pass (inference) never compute or
    hold the rest."""

    input_spikes: np.ndarray  # (T, n_in)
    membrane: np.ndarray  # (T, n_out), pre-threshold V at every step
    spikes: np.ndarray  # (T, n_out)


def forward_layer(input_spikes: np.ndarray, weights: np.ndarray, params: LIFLayerParams) -> LayerCache:
    """Run ``input_spikes`` (T, n_in) through one LIF layer with weight
    matrix ``weights`` (n_in, n_out). Returns the full cache; inference-only
    callers just read ``.spikes`` off it."""
    input_spikes = np.asarray(input_spikes, dtype=float)
    t_steps, n_out = input_spikes.shape[0], weights.shape[1]
    membrane = np.zeros((t_steps, n_out))
    spikes = np.zeros((t_steps, n_out))
    v = np.zeros(n_out)
    s_prev = np.zeros(n_out)
    decay = params.decay
    for t in range(t_steps):
        current = input_spikes[t] @ weights
        v = decay * v * (1.0 - s_prev) + current
        s = heaviside(v - params.v_th)
        membrane[t] = v
        spikes[t] = s
        s_prev = s
    return LayerCache(input_spikes=input_spikes, membrane=membrane, spikes=spikes)


def backward_layer(
    cache: LayerCache, weights: np.ndarray, params: LIFLayerParams, grad_spikes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """BPTT for one layer. ``grad_spikes`` is dL/dS, shape (T, n_out) --
    the gradient of the loss with respect to THIS layer's own spike output.
    Returns ``(grad_weights, grad_input_spikes)``: dL/dW (n_in, n_out), to
    update this layer, and dL/d(input_spikes) (T, n_in), to keep chaining
    backward into whatever layer produced those input spikes.
    """
    t_steps, n_out = grad_spikes.shape
    n_in = weights.shape[0]
    decay = params.decay
    grad_weights = np.zeros_like(weights)
    grad_input = np.zeros((t_steps, n_in))
    grad_v_next = np.zeros(n_out)  # dL/dV_t flowing in from step t+1
    for t in reversed(range(t_steps)):
        surrogate = fast_sigmoid_surrogate(cache.membrane[t] - params.v_th, k=params.surrogate_k)
        grad_v = grad_spikes[t] * surrogate + grad_v_next
        grad_weights += np.outer(cache.input_spikes[t], grad_v)
        grad_input[t] = grad_v @ weights.T
        s_prev = cache.spikes[t - 1] if t > 0 else np.zeros(n_out)
        # Gradient into V_{t-1} via the recurrence's `decay * V * (1 - s_prev)`
        # term; `(1 - s_prev)` itself is treated as constant (see module
        # docstring).
        grad_v_next = grad_v * decay * (1.0 - s_prev)
    return grad_weights, grad_input
