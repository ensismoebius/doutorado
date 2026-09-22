#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp"
#include "Meeting01GaFitness.hpp"
#include "Meeting01GaGenome.hpp"

namespace meeting01::ga
{

struct GaSearchConfig
{
    int population_size = 10;
    int generations = 8;
    double crossover_prob = 0.9;
    double mutation_prob = 0.2;
    int tournament_k = 2;
    unsigned int seed = 0; // 0 -> derive from base_seed passed to run_ga_search
    // How many seeds each final Pareto-front member is scored on before the winner is
    // picked. The search itself scores one seed per genome (cheap), which means the best
    // of ~90 noisy scores is partly selected on seed luck — the winner's curse. Re-scoring
    // only the front costs |front| * (winner_seeds - 1) extra trainings and makes the
    // published architecture a mean over seeds rather than a lucky draw. 1 disables it.
    int winner_seeds = 3;
    GenomeBounds bounds;

    // Checkpointing (two-layer, Meeting01GaCheckpoint.hpp). Empty results_dir disables it.
    std::string results_dir;
    std::string run_tag;
    int checkpoint_every_generations = 1;
};

struct GaSearchResult
{
    std::vector<Meeting01GaIndividual> final_population;
    // Feasible rank-0 individuals of the final population, sorted by (val_mse, then
    // inference_cost) — reversed from paraconsistentGA's (cost, then quality) ordering
    // because meeting01's primary published metric is reconstruction quality, not cost.
    std::vector<Meeting01GaIndividual> pareto_front;
    std::vector<Meeting01GaIndividual> history; // every distinct genome ever evaluated
    int generations_run = 0;
};

using EvalCallback = std::function<void(const Meeting01GaIndividual&)>;

// Runs NSGA-II (haploid — no diploid genetics; see Meeting01GaFitness.hpp for why no
// feasibility constraint is modeled either) over Meeting01Genome for one (dataset,
// fold, run_id). `split` is loaded once by the caller. Checkpointed via
// Meeting01GaCheckpoint.hpp iff ga_cfg.results_dir is non-empty.
auto run_ga_search(const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfig& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback& on_eval = nullptr) -> GaSearchResult;

// The individual the final retrain+test tail should use: result.pareto_front.front()
// (already ordered by val_mse then cost — reproduces today's min_element-by-val_mse
// grid-selection semantics as a special case). Throws if pareto_front is empty, which
// should be unreachable: population_size >= 1 and every individual here is
// unconditionally feasible.
auto pick_winner(const GaSearchResult& result) -> const Meeting01GaIndividual&;

} // namespace meeting01::ga
