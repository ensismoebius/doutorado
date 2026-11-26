/**
 * @file src/experiments/thesis/lib/include/FoldResult.hpp
 * @brief FoldResult struct (extracted from ThesisClassifiers.hpp).
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "core/training/EpochResult.hpp"

namespace thesis
{

// The inference energy of one test sample through the classifier (the extractor, the same for
// every classifier on one feature set, is not included). Measured on the fold's test pass:
// Meeting01Energy's ledger (Lemaire et al. 2022 at Horowitz 2014 prices, memory-aware).
struct InferenceEnergy
{
    double nj = 0.0;             // primary: memory-aware, a repeated input computed once
    double ops_nj = 0.0;         // its arithmetic part
    double mem_nj = 0.0;         // its memory part
    double nj_as_executed = 0.0; // the DSNN's first layer at every step (= nj for the ANN)
    double nj_sram8k = 0.0;      // every access priced at one SRAM size (sensitivity)
    double nj_sram32k = 0.0;
    double nj_sram1m = 0.0;
    double fp_macs = 0.0;
    double fp_adds = 0.0;
    double accesses = 0.0;
    double spikes = 0.0; // spikes per sample (DSNN); 0 for the ANN
};

// One extractor candidate of the per-fold protocol (Thesis-Per-Fold-Protocol.md §6), ranked by
// D_pen on the fold's training rows only.
struct ExtractorRank
{
    std::string label;
    double d_penalized = 0.0;
};

// One point of the hyperparameter grid, scored on the inner folds of one outer fold.
struct GridPoint
{
    int hidden_width = 0;
    int residual_blocks = 0; // layer_spec residual:<residual_blocks + 1>
    float learning_rate = 0.0f;
    float weight_decay = 0.0f;
    std::size_t param_count = 0;
    std::vector<double> inner_eer; // one per inner fold, in fold order
    double mean_inner_eer = 0.0;
};

// What the per-fold protocol chose inside one outer fold, with everything it chose among, so the
// stability of the choices is reported and not hidden.
struct FoldSelection
{
    std::vector<ExtractorRank> extractors; // in profile order
    std::size_t extractor_index = 0;       // the winner (lowest D_pen, first on ties)
    std::vector<GridPoint> grid;           // empty for the cosine control (nothing to tune)
    std::size_t grid_index = 0;            // the winner (lowest mean inner EER); grid only
    std::size_t param_count = 0;           // the refitted winner's; 0 for the cosine control
    double fold_ms = 0.0; // wall-clock of the whole fold: extraction, ranking, search, refit, test
};

// Per-fold classification result.
struct FoldResult
{
    int fold = 0;
    double accuracy = 0.0;
    double eer = 0.0;
    double loss = 0.0;
    double f1 = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double specificity = 0.0; // macro TN/(TN+FP)
    double auc = 0.0;
    std::string model_path;

    // Per-run stats (mirrors what the Meeting01/Meeting01 pipeline records per run).
    double train_ms = 0.0; // wall-clock to produce this fold's model
    double infer_ms = 0.0; // wall-clock to score the held-out test fold
    // Learning curve of the fold's final model: per-epoch train/val loss, epoch
    // time, and (SNN) mean spike rate + synaptic-operation count. Empty when the
    // trainer produced no epochs (e.g. a cache hit).
    std::vector<nn::training::EpochResult> history;
    // The test pass's energy per sample; nullopt for the cosine control (no network).
    std::optional<InferenceEnergy> energy;
    // The per-fold protocol's choices for this fold; nullopt in a fixed-configuration run.
    std::optional<FoldSelection> selection;
};

} // namespace thesis
