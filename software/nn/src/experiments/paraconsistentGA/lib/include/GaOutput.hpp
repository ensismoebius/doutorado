#pragma once

#include <string>
#include <vector>

#include "GaConfig.hpp"
#include "GaFitness.hpp"
#include "GaNsga2.hpp"

namespace pga
{

// Per-individual log (ga.md §6). One row per distinct evaluated genome, columns:
// generation, genome fields, alpha, beta, g1, g2, d_truth, d_penalized_mean/std,
// latent_activity, param_count, inference_cost, est_latency_ms, feasible.
// Path: results_dir/pga_{run_tag}_individuals.csv
void write_individuals_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<Individual>& history);

// Final Pareto front (ga.md §6/§7), queryable JSON: run metadata (population,
// modality, constraints, calibration flags) + one object per front member.
// Path: results_dir/pga_{run_tag}_pareto.json
void write_pareto_front_json(const std::string& results_dir,
    const std::string& run_tag,
    const GaConfig& cfg,
    const GaResult& result);

} // namespace pga
