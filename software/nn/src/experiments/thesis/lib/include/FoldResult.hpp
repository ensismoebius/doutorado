/**
 * @file src/experiments/thesis/lib/include/FoldResult.hpp
 * @brief FoldResult struct (extracted from ThesisClassifiers.hpp).
 */

#pragma once

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
};

} // namespace thesis
