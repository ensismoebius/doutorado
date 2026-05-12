/**
 * @file inference_tests.cpp
 * @brief Deterministic approximations for common inference tests.
 */

#include "statistics/inference_tests.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace statistics
{
namespace
{
auto normal_cdf(float x) -> float
{
    return 0.5f * std::erfc(-x / std::sqrt(2.0f));
}
} // namespace

auto cohens_d(const std::vector<float>& a, const std::vector<float>& b) -> float
{
    if (a.empty() || b.empty()) return 0.0f;

    const float mean_a = std::accumulate(a.begin(), a.end(), 0.0f) / static_cast<float>(a.size());
    const float mean_b = std::accumulate(b.begin(), b.end(), 0.0f) / static_cast<float>(b.size());

    float var_a = 0.0f;
    for (float v : a) var_a += (v - mean_a) * (v - mean_a);
    var_a /= static_cast<float>(std::max<std::size_t>(1, a.size() - 1));

    float var_b = 0.0f;
    for (float v : b) var_b += (v - mean_b) * (v - mean_b);
    var_b /= static_cast<float>(std::max<std::size_t>(1, b.size() - 1));

    const float pooled = std::sqrt((var_a + var_b) * 0.5f);
    if (pooled <= 1e-12f) return 0.0f;
    return (mean_a - mean_b) / pooled;
}

auto t_test_pvalue_approx(const std::vector<float>& a, const std::vector<float>& b) -> float
{
    if (a.size() < 2 || b.size() < 2) return 1.0f;

    const float mean_a = std::accumulate(a.begin(), a.end(), 0.0f) / static_cast<float>(a.size());
    const float mean_b = std::accumulate(b.begin(), b.end(), 0.0f) / static_cast<float>(b.size());

    float var_a = 0.0f;
    for (float v : a) var_a += (v - mean_a) * (v - mean_a);
    var_a /= static_cast<float>(a.size() - 1);

    float var_b = 0.0f;
    for (float v : b) var_b += (v - mean_b) * (v - mean_b);
    var_b /= static_cast<float>(b.size() - 1);

    const float se =
        std::sqrt(var_a / static_cast<float>(a.size()) + var_b / static_cast<float>(b.size()));
    if (se <= 1e-12f) return 1.0f;

    const float t = (mean_a - mean_b) / se;
    return 2.0f * (1.0f - normal_cdf(std::fabs(t)));
}

auto wilcoxon_signed_rank_pvalue_approx(const std::vector<float>& a, const std::vector<float>& b)
    -> float
{
    if (a.size() != b.size() || a.empty()) return 1.0f;

    struct DiffItem
    {
        float abs_diff;
        float sign;
    };

    std::vector<DiffItem> diffs;
    diffs.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float d = a[i] - b[i];
        if (std::fabs(d) <= 1e-12f) continue;
        diffs.push_back(DiffItem{std::fabs(d), d > 0.0f ? 1.0f : -1.0f});
    }
    if (diffs.empty()) return 1.0f;

    std::sort(diffs.begin(),
        diffs.end(),
        [](const DiffItem& lhs, const DiffItem& rhs) { return lhs.abs_diff < rhs.abs_diff; });

    float rank = 1.0f;
    float w_pos = 0.0f;
    float w_neg = 0.0f;
    for (const auto& item : diffs)
    {
        if (item.sign > 0.0f)
            w_pos += rank;
        else
            w_neg += rank;
        rank += 1.0f;
    }

    const float w = std::min(w_pos, w_neg);
    const float n = static_cast<float>(diffs.size());
    const float mean_w = n * (n + 1.0f) / 4.0f;
    const float std_w = std::sqrt(n * (n + 1.0f) * (2.0f * n + 1.0f) / 24.0f);
    if (std_w <= 1e-12f) return 1.0f;

    const float z = (w - mean_w) / std_w;
    return 2.0f * (1.0f - normal_cdf(std::fabs(z)));
}

} // namespace statistics