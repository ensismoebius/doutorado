import numpy as np
import pytest

from efficient_nn_lab.paraconsistent.demos.ga_pareto import dominance_example
from efficient_nn_lab.paraconsistent.ga_synthetic import (
    constrained_dominates,
    constraint_violation,
    dominates,
    first_front_mask,
    synthetic_generations,
)
from efficient_nn_lab.paraconsistent.metrics import CONTRADICTION_PENALTY, d_penalized, d_truth, g1_g2, score_point


# -- metrics: verified against software/nn's paraconsistent_ga_gtest reference
# case and .wiki/Core/Paraconsistent.md's own worked numbers --------------
def test_contradiction_penalty_is_two_minus_sqrt2():
    assert CONTRADICTION_PENALTY == pytest.approx(2.0 - 2.0**0.5, abs=1e-15)


def test_reference_case_alpha_092_beta_0075():
    point = score_point(alpha=0.92, beta=0.075)
    assert point.g1 == pytest.approx(0.845, abs=1e-9)
    assert point.g2 == pytest.approx(-0.005, abs=1e-9)
    assert point.d_truth == pytest.approx(0.1551, abs=1e-4)
    assert point.d_penalized == pytest.approx(0.1580, abs=1e-4)


@pytest.mark.parametrize(
    "g1,g2",
    [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
)
def test_non_truth_vertices_score_two_under_d_penalized_except_truth(g1, g2):
    score = d_penalized(g1, g2)
    if (g1, g2) == (1.0, 0.0):
        assert score == pytest.approx(0.0, abs=1e-9)
    else:
        assert score == pytest.approx(2.0, abs=1e-9)


def test_dead_extractor_exploit_and_fix():
    """alpha=beta=1 (a collapsed/"dead" feature extractor) scores a
    deceptively good D_truth but is correctly demoted by D_penalized."""
    g1, g2 = g1_g2(alpha=1.0, beta=1.0)
    assert d_truth(g1, g2) == pytest.approx(2.0**0.5, abs=1e-9)
    assert d_penalized(g1, g2) == pytest.approx(2.0, abs=1e-9)


# -- synthetic GA generations (illustrative, not real search output) ------
def test_synthetic_generations_shape():
    gens = synthetic_generations(n_generations=3, pop_size=6)
    assert len(gens) == 3
    assert all(pop.cost.shape == (6,) and pop.d_penalized.shape == (6,) for pop in gens)


def test_synthetic_generations_deterministic():
    a = synthetic_generations(n_generations=2, pop_size=4)
    b = synthetic_generations(n_generations=2, pop_size=4)
    assert all(np.array_equal(p.d_penalized, q.d_penalized) for p, q in zip(a, b))


def test_generation_zero_does_not_depend_on_the_ceiling():
    """The slider only changes where the population migrates to -- the
    dominance example on generation 0 must stay the same cloud."""
    low, high = synthetic_generations(latency_ceiling=0.3), synthetic_generations(latency_ceiling=1.2)
    assert np.array_equal(low[0].cost, high[0].cost)
    assert np.array_equal(low[0].d_penalized, high[0].d_penalized)


@pytest.mark.parametrize("ceiling", [0.3, 0.7, 1.2])
def test_last_generation_fits_under_the_ceiling(ceiling):
    last = synthetic_generations(latency_ceiling=ceiling)[-1]
    assert np.all(constraint_violation(last.cost, ceiling) == 0)


def test_constrained_dominance_follows_debs_three_cases():
    # 0: feasible but bad; 1: infeasible with a great D; 2: barely infeasible; 3: feasible, better than 0
    cost = np.array([0.5, 0.9, 0.75, 0.4])
    d = np.array([1.5, 0.1, 0.2, 1.0])
    violation = constraint_violation(cost, 0.7)
    assert constrained_dominates(0, 1, cost, d, violation)  # feasible beats infeasible, whatever D says
    assert not constrained_dominates(1, 0, cost, d, violation)
    assert constrained_dominates(2, 1, cost, d, violation)  # both infeasible: smaller violation wins
    assert constrained_dominates(3, 0, cost, d, violation)  # both feasible: plain Pareto
    assert not constrained_dominates(0, 3, cost, d, violation)


def test_front_is_feasible_and_non_dominated():
    pop = synthetic_generations(latency_ceiling=0.7)[0]
    violation = constraint_violation(pop.cost, 0.7)
    front = first_front_mask(pop.cost, pop.d_penalized, violation)
    assert front.any() and not np.any(front & (violation > 0))
    members = np.flatnonzero(front)
    for i in members:
        for j in members:
            assert not dominates(pop.cost[j], pop.d_penalized[j], pop.cost[i], pop.d_penalized[i])


def test_dominance_example_says_what_the_screen_says():
    """A dominates B; A and C are both on the front and neither dominates
    the other -- the exact claims the dominance step prints."""
    pop = synthetic_generations()[0]
    a, b, c = dominance_example(pop)
    cost, d = pop.cost, pop.d_penalized
    front = first_front_mask(cost, d)
    assert front[a] and front[c] and not front[b]
    assert cost[a] < cost[b] and d[a] < d[b]
    assert not dominates(cost[a], d[a], cost[c], d[c]) and not dominates(cost[c], d[c], cost[a], d[a])
