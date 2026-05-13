#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "E05Config.hpp"
#include "E05Dataset.hpp"

namespace e05
{

// Per-fold classification result.
struct FoldResult
{
    int fold = 0;
    double accuracy = 0.0;
    double eer = 0.0;
    double loss = 0.0;
};

// Full classification result over nested CV.
struct ClassificationResult
{
    std::string feature_set_label;
    std::string classifier_type;
    std::string text_mode;
    std::vector<FoldResult> outer_folds;
    double mean_accuracy = 0.0;
    double mean_eer = 0.0;
    double std_accuracy = 0.0;
};

// Train and evaluate the configured classifier using nested k-fold CV.
// feature_vectors: one vector per sample (aligned with view.samples).
// global_bar_id / global_completed: optional ProgressManager bar updated after each outer fold.
auto run_classifier(const E05DatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const E05Config& cfg,
    uint32_t global_bar_id = 0,
    int* global_completed = nullptr) -> ClassificationResult;

// Compute mean and std over fold accuracies.
void compute_aggregate_stats(ClassificationResult& result);

} // namespace e05
