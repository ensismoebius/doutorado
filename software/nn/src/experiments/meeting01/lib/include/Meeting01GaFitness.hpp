#pragma once

#include <cstddef>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp"
#include "Meeting01GaGenome.hpp"

namespace meeting01::ga
{

// A fully-evaluated GA individual, generic over the genome type G. Satisfies
// ga::Nsga2Scored (include/ga/Nsga2Core.hpp): feasible/constraint_violation/
// objectives/rank/crowding. Every family's architecture search (SNN via Genome,
// LSTM/GRU via RecurrentGenome, Transformer via TransformerGenome —
// Meeting01RecurrentGaGenome.hpp / Meeting01TransformerGaGenome.hpp) uses THIS same
// template instead of a hand-copied struct per family; only the genome shape differs,
// and `genome_type` lets the generic search driver (Meeting01GaSearch.hpp) and
// checkpoint layer (Meeting01GaCheckpoint.hpp) name it without being told explicitly.
//
// No feasibility constraint is modeled: unlike paraconsistentGA's latent-collapse
// guard (a real, previously-observed failure mode of the paraconsistent scoring
// pipeline), meeting01's objective is plain reconstruction MSE, which has no known
// degenerate false optimum — a collapsed/mean-predicting model scores WORSE on MSE,
// not artificially better. `feasible` stays true and `constraint_violation` stays 0.0
// for every individual; NSGA-II's constrained-dominance branch degenerates to plain
// Pareto dominance, which is the correct behavior here. Inventing a threshold-based
// "collapse guard" with no grounding in this pipeline's actual failure modes would be
// exactly the fabricated-recovery pattern CLAUDE.md's no-fallbacks rule forbids.
template <typename G>
struct GaIndividualT
{
    using genome_type = G;

    G genome{};

    float val_mse = 0.0f;
    std::size_t param_count = 0;
    std::size_t inference_cost = 0; // family-specific MAC estimate of this genome

    bool feasible = true;
    double constraint_violation = 0.0;

    std::vector<double> objectives; // {val_mse, inference_cost} — both minimized
    int rank = 0;
    double crowding = 0.0;

    int born_generation = -1;
};

// The SNN individual, kept under its original name — every existing caller
// (Meeting01GaSearch.*, Meeting01GaCheckpoint.*, Meeting01Experiment.cpp,
// meeting01_ga_gtest.cpp) refers to this type name unchanged.
using Meeting01GaIndividual = GaIndividualT<Genome>;

// Train this individual's genome on split.train_samples, score it on split.val_samples
// (exactly what run_snn_combo does today for one grid cell), and fill every field of
// `ind`. `progress_index`/`progress_total` are cosmetic only (fed to
// train_with_early_stopping_snn's progress-log context) — they do not affect training.
void evaluate_individual(Meeting01GaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total);

} // namespace meeting01::ga
