/**
 * @file inference_tests.cpp
 * @brief Deterministic small-sample inference tests.
 *
 * Corrections (audit M-3):
 *  - t-test p-value uses the Student-t CDF (regularised incomplete beta), not a
 *    normal approximation, and Welch–Satterthwaite degrees of freedom.
 *  - Wilcoxon signed-rank uses average ranks for ties, the tie-corrected
 *    variance, and a continuity correction.
 *  - Cohen's d uses the (n-1)-weighted pooled standard deviation.
 *
 * Note: cross-validation fold metrics are NOT independent (overlapping training
 * sets), so these tests remain approximate when applied across folds. Prefer a
 * corrected resampled-CV test (Nadeau & Bengio 2003) for fold-wise comparisons.
 */

#include "statistics/inference_tests.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace statistics
{
namespace
{
auto normal_cdf(double x) -> double
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Continued-fraction expansion for the incomplete beta function (Numerical
// Recipes, betacf). Converges for x < (a+1)/(a+b+2).
auto betacf(double a, double b, double x) -> double
{
    constexpr int kMaxIter = 200;
    constexpr double kEps = 3.0e-12;
    constexpr double kFpMin = 1.0e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= kMaxIter; ++m)
    {
        const double em = static_cast<double>(m);
        const double m2 = 2.0 * em;

        double aa = em * (b - em) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + em) * (qab + em) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return h;
}

// Regularised incomplete beta function I_x(a, b).
auto betai(double a, double b, double x) -> double
{
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    const double ln_beta = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    const double front = std::exp(ln_beta + a * std::log(x) + b * std::log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0))
        return front * betacf(a, b, x) / a;
    return 1.0 - front * betacf(b, a, 1.0 - x) / b;
}

// Two-sided p-value for a Student-t statistic with df degrees of freedom.
auto t_two_sided_pvalue(double t, double df) -> double
{
    if (df <= 0.0) return 1.0;
    const double x = df / (df + t * t);
    return betai(0.5 * df, 0.5, x); // = P(|T| > |t|)
}
} // namespace

auto cohens_d(const std::vector<float>& a, const std::vector<float>& b) -> float
{
    if (a.empty() || b.empty()) return 0.0f;

    const auto na = static_cast<double>(a.size());
    const auto nb = static_cast<double>(b.size());

    const double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / na;
    const double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / nb;

    double ss_a = 0.0;
    for (float v : a) ss_a += (static_cast<double>(v) - mean_a) * (static_cast<double>(v) - mean_a);
    double ss_b = 0.0;
    for (float v : b) ss_b += (static_cast<double>(v) - mean_b) * (static_cast<double>(v) - mean_b);

    // (n-1)-weighted pooled standard deviation: sqrt((SS_a + SS_b)/(na+nb-2)).
    const double dof = na + nb - 2.0;
    if (dof <= 0.0) return 0.0f;
    const double pooled = std::sqrt((ss_a + ss_b) / dof);
    if (pooled <= 1e-12) return 0.0f;
    return static_cast<float>((mean_a - mean_b) / pooled);
}

auto t_test_pvalue_approx(const std::vector<float>& a, const std::vector<float>& b) -> float
{
    if (a.size() < 2 || b.size() < 2) return 1.0f;

    const auto na = static_cast<double>(a.size());
    const auto nb = static_cast<double>(b.size());

    const double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / na;
    const double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / nb;

    double var_a = 0.0;
    for (float v : a) var_a += (static_cast<double>(v) - mean_a) * (static_cast<double>(v) - mean_a);
    var_a /= (na - 1.0);

    double var_b = 0.0;
    for (float v : b) var_b += (static_cast<double>(v) - mean_b) * (static_cast<double>(v) - mean_b);
    var_b /= (nb - 1.0);

    const double sa = var_a / na;
    const double sb = var_b / nb;
    const double se = std::sqrt(sa + sb);
    if (se <= 1e-12) return 1.0f;

    const double t = (mean_a - mean_b) / se;

    // Welch–Satterthwaite degrees of freedom.
    const double denom = (sa * sa) / (na - 1.0) + (sb * sb) / (nb - 1.0);
    const double df = (denom > 0.0) ? (sa + sb) * (sa + sb) / denom : (na + nb - 2.0);

    return static_cast<float>(t_two_sided_pvalue(std::fabs(t), df));
}

auto wilcoxon_signed_rank_pvalue_approx(const std::vector<float>& a, const std::vector<float>& b)
    -> float
{
    if (a.size() != b.size() || a.empty()) return 1.0f;

    struct DiffItem
    {
        double abs_diff;
        double sign;
    };

    std::vector<DiffItem> diffs;
    diffs.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        if (std::fabs(d) <= 1e-12) continue; // drop zero differences
        diffs.push_back(DiffItem{std::fabs(d), d > 0.0 ? 1.0 : -1.0});
    }
    if (diffs.empty()) return 1.0f;

    std::sort(diffs.begin(), diffs.end(),
        [](const DiffItem& lhs, const DiffItem& rhs) { return lhs.abs_diff < rhs.abs_diff; });

    const std::size_t n = diffs.size();

    // Assign average ranks to tied |differences| and accumulate the tie
    // correction term Σ(t³ - t) used in the variance.
    std::vector<double> ranks(n, 0.0);
    double tie_correction = 0.0;
    std::size_t i = 0;
    while (i < n)
    {
        std::size_t j = i + 1;
        while (j < n && std::fabs(diffs[j].abs_diff - diffs[i].abs_diff) <= 1e-12) ++j;
        const std::size_t group = j - i;                       // tie-group size
        const double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) ranks[k] = avg_rank;
        const double tg = static_cast<double>(group);
        tie_correction += tg * tg * tg - tg;
        i = j;
    }

    double w_pos = 0.0;
    double w_neg = 0.0;
    for (std::size_t k = 0; k < n; ++k)
    {
        if (diffs[k].sign > 0.0) w_pos += ranks[k];
        else                     w_neg += ranks[k];
    }

    const double w = std::min(w_pos, w_neg);
    const double nn = static_cast<double>(n);
    const double mean_w = nn * (nn + 1.0) / 4.0;
    // Tie-corrected variance: n(n+1)(2n+1)/24 - Σ(t³-t)/48.
    const double var_w = nn * (nn + 1.0) * (2.0 * nn + 1.0) / 24.0 - tie_correction / 48.0;
    if (var_w <= 1e-12) return 1.0f;
    const double std_w = std::sqrt(var_w);

    // z with continuity correction.
    const double z = (std::fabs(w - mean_w) - 0.5) / std_w;
    if (z <= 0.0) return 1.0f;
    return static_cast<float>(2.0 * (1.0 - normal_cdf(z)));
}

} // namespace statistics
