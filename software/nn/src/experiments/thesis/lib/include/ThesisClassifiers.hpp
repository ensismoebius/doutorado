#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ClassificationResult.hpp"
#include "FoldResult.hpp"
#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "core/training/EpochResult.hpp"
#include "statistics/eer_scorer.hpp"

namespace thesis
{

// Train and evaluate the configured classifier using nested k-fold CV.
// feature_vectors: one vector per sample (aligned with view.samples).
// eer_scorer:      pluggable EER strategy; nullptr → GenuineImpostorEERScorer (SOTA default).
// global_bar_id / global_completed: optional ProgressManager bar updated after each outer fold.
auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer* eer_scorer = nullptr,
    uint32_t global_bar_id = 0,
    int* global_completed = nullptr) -> ClassificationResult;

// Compute mean and std over fold accuracies.
void compute_aggregate_stats(ClassificationResult& result);

} // namespace thesis
