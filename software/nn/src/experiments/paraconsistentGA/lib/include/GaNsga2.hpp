#pragma once

#include <functional>
#include <vector>

#include "GaConfig.hpp"
#include "GaFitness.hpp"
#include "ThesisDataset.hpp"

namespace pga
{

// Constrained Pareto dominance (Deb 2002). Both objectives are minimized.
//   feasible vs infeasible  → feasible dominates
//   both infeasible         → smaller constraint_violation dominates
//   both feasible           → standard Pareto dominance on objectives
// Requires objectives + feasibility already filled (evaluated individuals).
bool constrained_dominates(const Individual& a, const Individual& b);

// Fast non-dominated sort. Assigns .rank (0 = best front) to every individual and
// returns the fronts as index lists into `pop`.
std::vector<std::vector<int>> fast_non_dominated_sort(std::vector<Individual>& pop);

// Crowding-distance assignment within one front (indices into `pop`). Writes
// .crowding on each; boundary points get +inf so extremes are always preserved.
void assign_crowding_distance(std::vector<Individual>& pop, const std::vector<int>& front);

// Result of a full GA run.
struct GaResult
{
    std::vector<Individual> final_population; // last generation, ranked
    std::vector<Individual> pareto_front;     // feasible rank-0 individuals
    std::vector<Individual> history;          // every distinct individual ever evaluated
    int generations_run = 0;
};

// Called once per individual right after it is evaluated (for streaming logs).
using EvalCallback = std::function<void(const Individual&)>;

// Run NSGA-II on one population. Dataset is loaded once by the caller. Every genome
// is evaluated via evaluate_individual (which reuses the thesis AE + paraconsistent
// pipeline). `on_eval`, if set, fires for each newly evaluated individual.
GaResult run_nsga2(const thesis::ThesisDatasetView& view,
    const GaConfig& cfg,
    const EvalCallback& on_eval = nullptr);

} // namespace pga
