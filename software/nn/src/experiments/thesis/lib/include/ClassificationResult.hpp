/**
 * @file src/experiments/thesis/lib/include/ClassificationResult.hpp
 * @brief ClassificationResult struct (extracted from ThesisClassifiers.hpp).
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "FoldResult.hpp"

namespace thesis
{

// Full classification result over nested CV.
struct ClassificationResult
{
    std::string feature_set_label;
    std::string classifier_type;
    std::string text_mode;
    std::vector<FoldResult> outer_folds;
    double mean_accuracy = 0.0;
    double std_accuracy = 0.0;
    double ci95_accuracy = 0.0;
    double mean_f1 = 0.0;
    double std_f1 = 0.0;
    double mean_precision = 0.0;
    double mean_recall = 0.0;
    double mean_specificity = 0.0;
    double std_specificity = 0.0;
    double mean_eer = 0.0;
    double std_eer = 0.0;
    double ci95_eer = 0.0;
    double mean_auc = 0.0;
    double std_auc = 0.0;

    // Run-level cost / complexity stats (the Guayaquil-style "info about the run").
    std::size_t param_count = 0;  // trainable parameters in the classifier
    double mean_train_ms = 0.0;   // mean per-fold training wall-clock
    double mean_infer_ms = 0.0;   // mean per-fold inference wall-clock
    double mean_spike_rate = 0.0; // SNN: mean firing rate at last epoch (NaN→skipped)
    long long final_sops = 0LL;   // SNN: synaptic ops/forward pass at last epoch
};

} // namespace thesis
