#pragma once

#include <cstdint>
#include <vector>

#include "GaConfig.hpp"
#include "GaGenome.hpp"
#include "ThesisDataset.hpp"

namespace pga
{

// A fully-evaluated GA individual: genotype + all logged phenotype quantities
// (ga.md §6, per-individual log). Objectives/rank/crowding are filled by NSGA-II.
struct Individual
{
    Genome genome;

    // Paraconsistent quality, averaged over n_seeds (ga.md §5.5). alpha/beta/g1/g2/
    // d_truth are from the best (lowest-d_penalized) seed so degenerate cases stay
    // diagnosable per individual (ga.md §5.2 caution).
    double d_penalized_mean = 0.0;
    double d_penalized_std = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double g1 = 0.0;
    double g2 = 0.0;
    double d_truth = 0.0;

    // Latent-collapse guard input: mean per-dimension std of the latent vectors
    // (best seed). Small ⇒ collapsed/dead latent ⇒ infeasible (§3.3 proxy).
    double latent_activity = 0.0;

    // Secondary objective (structural proxy) and its ms translation.
    long param_count = 0;
    long inference_cost = 0;     // encoder MACs · time_steps
    double est_latency_ms = 0.0; // fixed_pipeline_cost + proxy·ns_per_mac/1e6

    bool evaluated = false;
    bool feasible = true;
    double constraint_violation = 0.0; // 0 when feasible; larger = worse (for ranking)

    // ── NSGA-II bookkeeping ──────────────────────────────────────────────────
    std::vector<double> objectives; // {d_penalized_mean, inference_cost} — minimize both
    int rank = 0;
    double crowding = 0.0;

    int born_generation = -1; // generation the genome first appeared (for logging)
};

// Cheap pre-training screen (ga.md §4/§89): true if the genome's structural cost
// estimate already blows the latency budget, so it can be discarded WITHOUT training.
bool exceeds_latency_budget(const Genome& g, const GaConfig& cfg);

// Evaluate one individual: for each of n_seeds, build the AE config from the genome,
// reuse thesis::extract_features (trains the AE, returns latent vectors) and
// thesis::score_feature_set (returns d_penalized). Fills every field of `ind` and
// applies the constrained-feasibility rules (latency + latent-collapse). Individuals
// failing the pre-screen are marked infeasible and NOT trained.
//
// `view` is the dataset loaded once by the caller (seed-independent).
void evaluate_individual(
    Individual& ind, const thesis::ThesisDatasetView& view, const GaConfig& cfg);

} // namespace pga
