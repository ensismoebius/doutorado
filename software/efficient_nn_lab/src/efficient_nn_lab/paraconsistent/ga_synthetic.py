"""Synthetic, illustrative NSGA-II populations for the ga_pareto demo.

Shaped like `software/nn`'s real `paraconsistentGA` experiment -- two
minimized objectives `{d_penalized_mean, inference_cost}`
(`lib/include/GaFitness.hpp`), a latency ceiling that makes individuals
infeasible, and Deb's constrained domination (`include/ga/Nsga2Core.hpp`
`constrained_dominates`, mirrored by :func:`constrained_dominates` here) --
but every number is generated for this demo, not read from a real run.

The population is a cloud, not a line: generation 0 is scattered above an
"ideal" trade-off curve (:func:`ideal_front`), so most of it is dominated,
as in a random initial population. Each later generation lowers the cloud
onto that curve and spreads it along the part of the cost axis that fits
under the latency ceiling -- the two effects NSGA-II's elitism and crowding
distance produce. Slot i in generation g+1 is drawn as "the descendant of
slot i in generation g" so the demo can glide between generations; that is a
visual simplification (the real algorithm replaces individuals, it never
moves one).

The bridge "the thesis's D_penalized as an architecture-search objective"
is `paraconsistentGA`'s, not the thesis monography's: the thesis ranks
feature-extraction configurations with D_penalized and never runs a genetic
search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efficient_nn_lab.core.math_utils import SEED

#: Normalized cost range the synthetic architectures span (stand-in for
#: encoder MACs x time_steps, which the real run counts in raw MACs).
COST_LOW, COST_HIGH = 0.08, 1.25
_D_CAP = 1.95  # D_penalized never exceeds 2 (the degenerate-vertex value)


@dataclass(frozen=True)
class Population:
    cost: np.ndarray  # objective 2 (minimize): normalized inference cost
    d_penalized: np.ndarray  # objective 1 (minimize): feature quality of the trained AE's latent


def ideal_front(cost: np.ndarray | float) -> np.ndarray | float:
    """The trade-off the synthetic search converges to: D_penalized falls as
    cost rises, with diminishing returns (doubling the cost buys less and
    less quality)."""
    return 0.2 + 0.075 / (np.asarray(cost) + 0.02)


def synthetic_generations(
    n_generations: int = 5,
    pop_size: int = 12,
    latency_ceiling: float = 0.7,
    seed: int = SEED,
) -> list[Population]:
    """One population per generation. Every random draw happens before the
    ceiling is used, so generation 0 is the same cloud for every ceiling --
    only where the population migrates to depends on it."""
    rng = np.random.RandomState(seed)
    # Stratified: one random cost per equal-width bin, so the cloud looks
    # random but no two architectures sit at (almost) the same cost -- a
    # near-tie would put a point "on the front" by a hair, which teaches
    # nothing about dominance.
    edges = np.linspace(COST_LOW, COST_HIGH, pop_size + 1)
    cost0 = rng.permutation(edges[:-1] + (edges[1:] - edges[:-1]) * rng.uniform(0.2, 0.8, pop_size))
    gap0 = rng.uniform(0.2, 1.0, pop_size)  # distance above the ideal curve
    gap_final = rng.uniform(0.0, 0.05, pop_size)

    # Final positions: evenly spread under the ceiling, in generation 0's cost
    # order (so no two descendants cross paths on the way there).
    high = max(COST_LOW + 0.05, min(COST_HIGH, latency_ceiling) - 0.02)
    cost_final = np.empty(pop_size)
    cost_final[np.argsort(cost0)] = np.linspace(COST_LOW, high, pop_size)

    populations = []
    for gen in range(n_generations):
        progress = gen / max(1, n_generations - 1)
        cost = cost0 + (cost_final - cost0) * progress
        gap = gap_final + (gap0 - gap_final) * (1.0 - progress) ** 1.5
        populations.append(Population(cost=cost, d_penalized=np.minimum(_D_CAP, ideal_front(cost) + gap)))
    return populations


def dominates(cost_a: float, d_a: float, cost_b: float, d_b: float) -> bool:
    """Plain Pareto dominance, both objectives minimized: a is no worse in
    either objective and strictly better in at least one."""
    return cost_a <= cost_b and d_a <= d_b and (cost_a < cost_b or d_a < d_b)


def constraint_violation(cost: np.ndarray, ceiling: float) -> np.ndarray:
    """0 when the individual fits under the ceiling; how far over it otherwise."""
    return np.maximum(0.0, np.asarray(cost, dtype=float) - ceiling)


def constrained_dominates(i: int, j: int, cost: np.ndarray, d: np.ndarray, violation: np.ndarray) -> bool:
    """Deb's rule, mirroring `ga::constrained_dominates`: feasible beats
    infeasible whatever the objectives say; between two infeasible ones the
    smaller violation wins; between two feasible ones, plain Pareto."""
    feasible_i, feasible_j = violation[i] <= 0.0, violation[j] <= 0.0
    if feasible_i != feasible_j:
        return bool(feasible_i)
    if not feasible_i:
        return bool(violation[i] < violation[j])
    return dominates(cost[i], d[i], cost[j], d[j])


def first_front_mask(cost: np.ndarray, d: np.ndarray, violation: np.ndarray | None = None) -> np.ndarray:
    """True for the individuals nobody constrained-dominates (NSGA-II rank 0).

    With ``violation`` omitted every individual is feasible and this is the
    plain Pareto front.
    """
    n = len(cost)
    v = np.zeros(n) if violation is None else np.asarray(violation, dtype=float)
    return np.array(
        [not any(constrained_dominates(j, i, cost, d, v) for j in range(n) if j != i) for i in range(n)],
        dtype=bool,
    )
