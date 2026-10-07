"""Paraconsistent logic metrics (da Costa paraconsistent feature engineering).

Pure functions mirroring `software/nn`'s thesis-novel contribution
(`include/paraconsistent/paraconsistent.hpp`, `ThesisParaconsistent.hpp`):
intraclass compactness (alpha) and interclass overlap (beta) map to a point
(G1, G2) on the "paraconsistent plane", and D_penalized is the metric
actually used to rank feature-extraction configurations before any
classifier is trained (software/nn's .wiki/Core/Paraconsistent.md).
"""

from __future__ import annotations

from dataclasses import dataclass

#: lambda = 2 - sqrt(2), chosen so Falsity/Ambiguity/Indefinition all score
#: exactly 2.0 under D_penalized while Truth scores 0 (software/nn's
#: ThesisParaconsistent.hpp::kContradictionPenalty, reproduced verbatim so
#: this demo's numbers stay bit-identical to the real one).
CONTRADICTION_PENALTY = 0.5857864376269049

#: The paraconsistent plane's 4 labeled vertices, as (g1, g2) pairs.
VERTICES: dict[str, tuple[float, float]] = {
    "Verdade": (1.0, 0.0),
    "Falsidade": (-1.0, 0.0),
    "Ambiguidade": (0.0, 1.0),
    "Indefinição": (0.0, -1.0),
}


@dataclass(frozen=True)
class ParaconsistentPoint:
    alpha: float
    beta: float
    g1: float
    g2: float
    d_truth: float
    d_penalized: float


def g1_g2(alpha: float, beta: float) -> tuple[float, float]:
    """(certainty, contradiction) degrees from (alpha, beta)."""
    return alpha - beta, alpha + beta - 1.0


def d_truth(g1: float, g2: float) -> float:
    """Euclidean distance from (g1, g2) to the Truth vertex (1, 0)."""
    return ((g1 - 1.0) ** 2 + g2**2) ** 0.5


def d_penalized(g1: float, g2: float) -> float:
    """D_truth plus a penalty on |contradiction| -- the metric actually used
    to rank feature sets, because D_truth alone is exploitable (see
    :func:`score_point`).
    """
    return d_truth(g1, g2) + CONTRADICTION_PENALTY * abs(g2)


def score_point(alpha: float, beta: float) -> ParaconsistentPoint:
    """Full (alpha, beta) -> plane point, including both distance metrics.

    D_truth alone is exploitable: a collapsed ("dead") feature extractor
    that emits the same output for every sample regardless of class lands
    exactly on the Ambiguity vertex (alpha=beta=1), scoring
    D_truth = sqrt(2) ~= 1.4142 -- which outranks any weak-but-real feature
    set with D_truth > sqrt(2) (alpha=0.2, beta=0.7 scores 1.5033), despite
    carrying zero class information. D_penalized fixes this: at Ambiguity,
    Falsity and Indefinition it scores exactly 2.0, the worst value on the
    plane, so a degenerate extractor can never outrank an informative one.
    """
    g1, g2 = g1_g2(alpha, beta)
    return ParaconsistentPoint(
        alpha=alpha,
        beta=beta,
        g1=g1,
        g2=g2,
        d_truth=d_truth(g1, g2),
        d_penalized=d_penalized(g1, g2),
    )
