#pragma once

#include <string>
#include <vector>

#include "E05Dataset.hpp"
#include "E05FeatureExtraction.hpp"

namespace e05
{

// Contradiction penalty weight for d_penalized (see below).
// Chosen as 2 - sqrt(2) so that the three non-Truth vertices of the
// paraconsistent plane are penalized equally: Falsity(-1,0), Ambiguity(0,1)
// and Indefinition(0,-1) all score exactly 2.0, while Truth(1,0) scores 0.
inline constexpr double kContradictionPenalty = 0.5857864376269049; // 2 - sqrt(2)

// Paraconsistent quality score for one feature set.
struct ParaconsistentScore
{
    std::string label;
    double alpha = 0.0;
    double beta = 0.0;
    double g1 = 0.0;      // certainty degree  (alpha - beta)
    double g2 = 0.0;      // contradiction degree (alpha + beta - 1)
    double d_truth = 0.0; // distance to Truth corner (1,0); smaller is better
    // Primary selection metric. d_truth alone is exploitable: a collapsed
    // ("dead") latent lands on the Ambiguity vertex (alpha=beta=1) and scores
    // a low d_truth (sqrt(2)) despite carrying no class information. This adds
    // a penalty on |g2| (the contradiction degree, whose two poles are exactly
    // the degenerate Ambiguity and Indefinition vertices), so those states are
    // scored as the worst possible. Smaller is better.
    //   d_penalized = d_truth + kContradictionPenalty * |g2|
    double d_penalized = 0.0;
};

// Rank all feature sets by d_penalized using the paraconsistent EPC/α/β metric.
// samples must be aligned with each FeatureSet::vectors.
// Returns scores sorted ascending by d_penalized (best first).
auto rank_feature_sets(const std::vector<E05Sample>& samples,
    const std::vector<FeatureSet>& feature_sets) -> std::vector<ParaconsistentScore>;

// Compute the paraconsistent score for a single feature set.
auto score_feature_set(const std::vector<E05Sample>& samples, const FeatureSet& fs)
    -> ParaconsistentScore;

} // namespace e05
