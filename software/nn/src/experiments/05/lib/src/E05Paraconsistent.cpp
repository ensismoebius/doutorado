#include "E05Paraconsistent.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "paraconsistent/paraconsistent.hpp"

namespace e05
{

auto score_feature_set(const std::vector<E05Sample>& samples,
    const FeatureSet& fs) -> ParaconsistentScore
{
    if (fs.vectors.empty() || samples.size() != fs.vectors.size())
        throw std::invalid_argument("E05Paraconsistent: empty or mismatched feature set");

    // Build speaker-keyed map: "subject_N" -> [ [feat_vec], ... ]
    std::map<std::string, std::vector<std::vector<double>>> class_map;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        std::string key = "subject_" + std::to_string(samples[i].subject_id);
        class_map[key].push_back(fs.vectors[i]);
    }

    // Trim to equal-sized classes (minimum count across speakers).
    size_t min_count = std::numeric_limits<size_t>::max();
    for (auto& [k, v] : class_map)
        min_count = std::min(min_count, v.size());
    for (auto& [k, v] : class_map)
        v.resize(min_count);

    if (min_count == 0)
        throw std::runtime_error("E05Paraconsistent: no samples after trimming");

    unsigned int n_classes = static_cast<unsigned int>(class_map.size());
    unsigned int n_per_class = static_cast<unsigned int>(min_count);
    unsigned int feat_dim = static_cast<unsigned int>(fs.vectors[0].size());

    normalize_class_feature_vectors(n_classes, n_per_class, feat_dim, class_map);

    double alpha = calculate_alpha(n_classes, n_per_class, feat_dim, class_map);
    double beta  = calculate_beta(n_classes, n_per_class, feat_dim, class_map);
    double g1    = calculate_certainty_degree_g1(alpha, beta);
    double g2    = calculate_contradiction_degree_g2(alpha, beta);

    // Distance to Truth corner (1, 0) in (G1, G2) plane.
    double d_truth = std::sqrt((g1 - 1.0) * (g1 - 1.0) + g2 * g2);

    return {fs.label, alpha, beta, g1, g2, d_truth};
}

auto rank_feature_sets(const std::vector<E05Sample>& samples,
    const std::vector<FeatureSet>& feature_sets) -> std::vector<ParaconsistentScore>
{
    std::vector<ParaconsistentScore> scores;
    scores.reserve(feature_sets.size());

    for (const auto& fs : feature_sets)
    {
        if (fs.vectors.empty()) continue; // skip placeholder (autoencoder, not trained yet)
        scores.push_back(score_feature_set(samples, fs));
    }

    std::sort(scores.begin(), scores.end(),
        [](const ParaconsistentScore& a, const ParaconsistentScore& b)
        { return a.d_truth < b.d_truth; });

    return scores;
}

} // namespace e05
