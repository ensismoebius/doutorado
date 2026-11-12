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

/// Simulated dysphonia at test time: features extracted from the degraded voice, aligned
/// with view.samples. They replace the clean features of the TEST speakers only, scaled with
/// the scaler fitted on the clean training rows (the system was trained on healthy voices).
struct DegradedTestFeatures
{
    const std::vector<std::vector<double>>* vectors = nullptr;
    /// degradation.scope "probes": each test speaker's enrolment rows stay clean.
    bool enrol_clean = false;
    /// How many rows per speaker the scorer takes as enrolment: the FIRST ones in test
    /// order (GenuineImpostorEERScorer). Must equal the scorer's count.
    std::size_t enrol_per_speaker = 0;
};

// Train and evaluate the configured classifier using nested k-fold CV.
// feature_vectors: one vector per sample (aligned with view.samples).
// eer_scorer:      the EER/AUC strategy, required (there is no default scorer).
// global_bar_id / global_completed: optional ProgressManager bar updated after each outer fold.
auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer,
    uint32_t global_bar_id = 0,
    int* global_completed = nullptr,
    const DegradedTestFeatures* degraded = nullptr) -> ClassificationResult;

// Compute mean and std over fold accuracies.
void compute_aggregate_stats(ClassificationResult& result);

} // namespace thesis
