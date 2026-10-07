"""Threshold-dependent batch normalization (tdBN), didactic single-channel
version of `include/layers/spiking/ThresholdDependentBatchNorm.hpp`
(Zheng et al. 2021). Pools statistics over the whole pooled batch+time
population and rescales to spread alpha*V_th instead of unit variance, so
a predictable fraction of neurons sits near threshold regardless of depth
(software/nn's .wiki/Concepts/Threshold-Dependent-Batch-Normalization.md).
"""

from __future__ import annotations

import numpy as np

#: Matches ThresholdDependentBatchNorm.hpp's default numerical-stability
#: constant.
DEFAULT_EPS = 1e-5


def tdbn_transform(x: np.ndarray, v_th: float, alpha: float = 1.0, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Y = alpha * v_th * (X - mean) / sqrt(var + eps) -- gamma=1, beta=0.

    ``var`` is the biased (population, ddof=0) variance pooled over every
    value of ``x``, matching the C++ layer's 1/N formula (N = T*B pooled
    rows for one channel), not numpy's default sample variance.
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    var = x.var()
    x_hat = (x - mean) / np.sqrt(var + eps)
    return alpha * v_th * x_hat
