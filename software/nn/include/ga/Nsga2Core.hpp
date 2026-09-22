/**
 * @file include/ga/Nsga2Core.hpp
 * @brief Backend-agnostic NSGA-II core: constrained dominance, fast non-dominated
 *        sort, crowding-distance assignment (Deb et al. 2002; constrained dominance
 *        per Deb 2000). Extracted from paraconsistentGA's GaNsga2.cpp, which only
 *        ever touched five fields (feasible/constraint_violation/objectives/rank/
 *        crowding) — genuinely population-shape-agnostic, so it is shared here
 *        between paraconsistentGA and meeting01's GA rather than duplicated.
 */

#ifndef NN_GA_NSGA2_CORE_HPP
#define NN_GA_NSGA2_CORE_HPP

#include <algorithm>
#include <concepts>
#include <limits>
#include <vector>

namespace ga
{

// Any per-individual struct NSGA-II can rank: two minimized objectives (or more),
// a feasibility flag + violation magnitude for constrained dominance (Deb 2000), and
// rank/crowding slots the algorithm writes into.
template <typename T>
concept Nsga2Scored = requires(T& t, const T& ct) {
    { ct.feasible } -> std::convertible_to<bool>;
    { ct.constraint_violation } -> std::convertible_to<double>;
    { ct.objectives } -> std::convertible_to<const std::vector<double>&>;
    { t.rank } -> std::convertible_to<int&>;
    { t.crowding } -> std::convertible_to<double&>;
};

// Constrained Pareto dominance (Deb 2002). Both objectives are minimized.
//   feasible vs infeasible  -> feasible dominates
//   both infeasible         -> smaller constraint_violation dominates
//   both feasible           -> standard Pareto dominance on objectives
// Requires objectives + feasibility already filled (evaluated individuals).
template <Nsga2Scored T>
bool constrained_dominates(const T& a, const T& b)
{
    if (a.feasible != b.feasible) return a.feasible; // feasible dominates infeasible
    if (!a.feasible && !b.feasible)                  // both infeasible: less violation wins
        return a.constraint_violation < b.constraint_violation;

    // Both feasible: standard Pareto on minimized objectives.
    bool strictly_better = false;
    const std::size_t m = std::min(a.objectives.size(), b.objectives.size());
    for (std::size_t i = 0; i < m; ++i)
    {
        if (a.objectives[i] > b.objectives[i]) return false;
        if (a.objectives[i] < b.objectives[i]) strictly_better = true;
    }
    return strictly_better;
}

// Fast non-dominated sort. Assigns .rank (0 = best front) to every individual and
// returns the fronts as index lists into `pop`.
template <Nsga2Scored T>
std::vector<std::vector<int>> fast_non_dominated_sort(std::vector<T>& pop)
{
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> dominates(n);
    std::vector<int> dominated_count(n, 0);
    std::vector<std::vector<int>> fronts;
    fronts.emplace_back();

    for (int p = 0; p < n; ++p)
    {
        for (int q = 0; q < n; ++q)
        {
            if (p == q) continue;
            if (constrained_dominates(pop[p], pop[q]))
                dominates[p].push_back(q);
            else if (constrained_dominates(pop[q], pop[p]))
                ++dominated_count[p];
        }
        if (dominated_count[p] == 0)
        {
            pop[p].rank = 0;
            fronts[0].push_back(p);
        }
    }

    int fi = 0;
    while (!fronts[fi].empty())
    {
        std::vector<int> next;
        for (int p : fronts[fi])
            for (int q : dominates[p])
                if (--dominated_count[q] == 0)
                {
                    pop[q].rank = fi + 1;
                    next.push_back(q);
                }
        ++fi;
        fronts.push_back(std::move(next));
    }
    fronts.pop_back(); // last pushed front is empty
    return fronts;
}

// Crowding-distance assignment within one front (indices into `pop`). Writes
// .crowding on each; boundary points get +inf so extremes are always preserved.
template <Nsga2Scored T>
void assign_crowding_distance(std::vector<T>& pop, const std::vector<int>& front)
{
    const std::size_t l = front.size();
    for (int idx : front) pop[idx].crowding = 0.0;
    if (l == 0) return;
    if (l <= 2)
    {
        for (int idx : front) pop[idx].crowding = std::numeric_limits<double>::infinity();
        return;
    }

    const std::size_t n_obj = pop[front[0]].objectives.size();
    for (std::size_t m = 0; m < n_obj; ++m)
    {
        std::vector<int> order(front);
        std::sort(order.begin(),
            order.end(),
            [&](int a, int b) { return pop[a].objectives[m] < pop[b].objectives[m]; });

        pop[order.front()].crowding = std::numeric_limits<double>::infinity();
        pop[order.back()].crowding = std::numeric_limits<double>::infinity();

        const double lo = pop[order.front()].objectives[m];
        const double hi = pop[order.back()].objectives[m];
        const double range = hi - lo;
        if (range <= 0.0) continue;

        for (std::size_t i = 1; i + 1 < l; ++i)
        {
            if (pop[order[i]].crowding == std::numeric_limits<double>::infinity()) continue;
            pop[order[i]].crowding +=
                (pop[order[i + 1]].objectives[m] - pop[order[i - 1]].objectives[m]) / range;
        }
    }
}

} // namespace ga

#endif // NN_GA_NSGA2_CORE_HPP
