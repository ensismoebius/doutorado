/**
 * @file src/experiments/thesis/lib/include/FoldResult.hpp
 * @brief FoldResult struct (extracted from ThesisClassifiers.hpp).
 */

#pragma once

#include <string>
#include <vector>

#include "core/training/EpochResult.hpp"

namespace thesis
{

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

    // Per-run stats (mirrors what the Guayaquil/Guayaquil pipeline records per run).
    double train_ms = 0.0; // wall-clock to produce this fold's model
    double infer_ms = 0.0; // wall-clock to score the held-out test fold
    // Learning curve of the fold's final model: per-epoch train/val loss, epoch
    // time, and (SNN) mean spike rate + synaptic-operation count. Empty when the
    // trainer produced no epochs (e.g. a cache hit).
    std::vector<nn::training::EpochResult> history;
};

} // namespace thesis
