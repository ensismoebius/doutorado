"""Firing-rate regularization: a soft penalty pushing the mean firing rate
toward a target band [r_min, r_max], against dead and bursting layers.

"Mean" is ONE number per layer -- the average over every unit, frame and
sample of the layer's spike tensor -- so the push is the same for every
unit of the layer, and a layer mixing dead and bursting units can average
into the band and get no push at all.
Mirrors `SpikeCountLossImpl`'s `rate_reg_lambda`/`min_rate`/`max_rate`
(`include/layers/losses/SpikeCountLoss.hpp`), per software/nn's
.wiki/Concepts/Spike-Rate-Regularization.md.
"""

from __future__ import annotations

#: The class's own default guard-rail values (not a literature-tuned
#: target band -- see the wiki page's distinction between the two).
DEFAULT_MIN_RATE = 0.05
DEFAULT_MAX_RATE = 0.80


def rate_reg_loss(mean_rate: float, lambda_reg: float, r_min: float = DEFAULT_MIN_RATE, r_max: float = DEFAULT_MAX_RATE) -> float:
    """L_reg = lambda * [max(0, r_min - rho)^2 + max(0, rho - r_max)^2]."""
    below = max(0.0, r_min - mean_rate)
    above = max(0.0, mean_rate - r_max)
    return lambda_reg * (below**2 + above**2)


def rate_reg_push(mean_rate: float, lambda_reg: float, r_min: float = DEFAULT_MIN_RATE, r_max: float = DEFAULT_MAX_RATE) -> float:
    """How much, and in which direction, the regularizer pushes the rate.

    Returns ``-dL_reg/d(mean_rate)`` (the direction gradient DESCENT moves
    the rate in): positive below ``r_min`` (push up), negative above
    ``r_max`` (push down), zero inside the band.
    """
    clamped = min(max(mean_rate, r_min), r_max)
    grad = 2.0 * lambda_reg * (mean_rate - clamped)
    return 0.0 - grad  # not -grad: inside the band that would be IEEE -0.0, shown as "-0.000"
