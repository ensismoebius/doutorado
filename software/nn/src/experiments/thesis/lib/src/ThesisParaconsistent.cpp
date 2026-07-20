#include "ThesisParaconsistent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include "paraconsistent/paraconsistent.hpp"
#include "progress/ProgressManager.hpp"

namespace thesis
{

namespace
{
// Per-dimension min-max scaling of every feature vector into [0,1] across the
// whole sample set (audit M-1). The paraconsistent α/β metric is defined on
// commensurable features in [0,1]; the previous per-sample sum-1 normalization
// let large-magnitude descriptors (e.g. energy) dominate and could push signed
// descriptors (e.g. Teager) outside [0,1], breaking the α domain. Scaling each
// dimension independently across samples keeps every component in [0,1] and
// gives each descriptor comparable influence on the range-based metric.
std::vector<std::vector<double>> min_max_per_dim(const std::vector<std::vector<double>>& vectors)
{
    if (vectors.empty()) return {};
    const size_t dim = vectors[0].size();

    std::vector<double> mn(dim, std::numeric_limits<double>::max());
    std::vector<double> mx(dim, std::numeric_limits<double>::lowest());
    for (const auto& v : vectors)
        for (size_t d = 0; d < dim && d < v.size(); ++d)
        {
            mn[d] = std::min(mn[d], v[d]);
            mx[d] = std::max(mx[d], v[d]);
        }

    std::vector<std::vector<double>> scaled(vectors.size(), std::vector<double>(dim, 0.0));
    for (size_t i = 0; i < vectors.size(); ++i)
        for (size_t d = 0; d < dim && d < vectors[i].size(); ++d)
        {
            const double range = mx[d] - mn[d];
            scaled[i][d] = (range > 0.0) ? (vectors[i][d] - mn[d]) / range : 0.0;
        }
    return scaled;
}
} // namespace

auto score_feature_set(const std::vector<ThesisSample>& samples, const FeatureSet& fs)
    -> ParaconsistentScore
{
    if (fs.vectors.empty() || samples.size() != fs.vectors.size())
        throw std::invalid_argument("ThesisParaconsistent: empty or mismatched feature set");

    // Scale features per-dimension to [0,1] before building class groups.
    const auto scaled_vectors = min_max_per_dim(fs.vectors);

    // Build speaker-keyed map: "subject_N" -> [ [feat_vec], ... ]
    std::map<std::string, std::vector<std::vector<double>>> class_map;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        std::string key = "subject_" + std::to_string(samples[i].subject_id);
        class_map[key].push_back(scaled_vectors[i]);
    }

    // Trim to equal-sized classes (minimum count across speakers).
    size_t min_count = std::numeric_limits<size_t>::max();
    for (auto& [k, v] : class_map) min_count = std::min(min_count, v.size());
    for (auto& [k, v] : class_map) v.resize(min_count);

    if (min_count == 0) throw std::runtime_error("ThesisParaconsistent: no samples after trimming");

    unsigned int n_classes = static_cast<unsigned int>(class_map.size());
    unsigned int n_per_class = static_cast<unsigned int>(min_count);
    unsigned int feat_dim = static_cast<unsigned int>(fs.vectors[0].size());

    // Features already scaled to [0,1] per dimension above (audit M-1);
    // no further per-sample normalization is applied.
    double alpha = calculate_alpha(n_classes, n_per_class, feat_dim, class_map);
    double beta = calculate_beta(n_classes, n_per_class, feat_dim, class_map);
    double g1 = calculate_certainty_degree_g1(alpha, beta);
    double g2 = calculate_contradiction_degree_g2(alpha, beta);

    // Distance to Truth corner (1, 0) in (G1, G2) plane.
    double d_truth = std::sqrt((g1 - 1.0) * (g1 - 1.0) + g2 * g2);

    // Contradiction-penalized truth distance (primary selection metric).
    // |g2| is the proximity to the degenerate Ambiguity/Indefinition axis;
    // penalizing it makes a collapsed latent (alpha=beta=1) the worst case
    // instead of a false winner. See ParaconsistentScore::d_penalized.
    double d_penalized = d_truth + kContradictionPenalty * std::abs(g2);

    return {fs.label, alpha, beta, g1, g2, d_truth, d_penalized};
}

auto rank_feature_sets(const std::vector<ThesisSample>& samples,
    const std::vector<FeatureSet>& feature_sets) -> std::vector<ParaconsistentScore>
{
    std::vector<ParaconsistentScore> scores;
    scores.reserve(feature_sets.size());

    uint32_t rank_bar = nn::progress::ProgressManager::instance().create_bar(
        "Paraconsistent ranking", static_cast<float>(feature_sets.size()));
    nn::progress::ProgressManager::instance().set_description(
        rank_bar, "Computing α/β/D_penalized per feature set");

    int ranked = 0;
    for (const auto& fs : feature_sets)
    {
        if (!fs.vectors.empty()) scores.push_back(score_feature_set(samples, fs));
        nn::progress::ProgressManager::instance().update_bar(
            rank_bar, static_cast<float>(++ranked));
    }

    nn::progress::ProgressManager::instance().complete_bar(rank_bar);

    std::sort(scores.begin(),
        scores.end(),
        [](const ParaconsistentScore& a, const ParaconsistentScore& b)
        { return a.d_penalized < b.d_penalized; });

    return scores;
}

} // namespace thesis
