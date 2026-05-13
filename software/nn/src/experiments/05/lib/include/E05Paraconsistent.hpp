#pragma once

#include <string>
#include <vector>

#include "E05Dataset.hpp"
#include "E05FeatureExtraction.hpp"

namespace e05
{

// Paraconsistent quality score for one feature set.
struct ParaconsistentScore
{
    std::string label;
    double alpha = 0.0;
    double beta = 0.0;
    double g1 = 0.0;  // certainty degree
    double g2 = 0.0;  // contradiction degree
    double d_truth = 0.0; // distance to Truth corner; smaller is better
};

// Rank all feature sets by D_truth using the paraconsistent EPC/α/β metric.
// samples must be aligned with each FeatureSet::vectors.
// Returns scores sorted ascending by d_truth (best first).
auto rank_feature_sets(const std::vector<E05Sample>& samples,
    const std::vector<FeatureSet>& feature_sets) -> std::vector<ParaconsistentScore>;

// Compute the paraconsistent score for a single feature set.
auto score_feature_set(const std::vector<E05Sample>& samples,
    const FeatureSet& fs) -> ParaconsistentScore;

} // namespace e05
