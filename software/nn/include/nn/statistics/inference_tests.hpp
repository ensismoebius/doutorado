/**
 * @file inference_tests.hpp
 * @brief Lightweight deterministic statistical inference helpers.
 */

#ifndef NN_STATISTICS_INFERENCE_TESTS_HPP
#define NN_STATISTICS_INFERENCE_TESTS_HPP

#include <vector>

namespace statistics
{

/**
 * Compute Cohen's d effect size for two samples.
 * Returns 0 when either input is empty or pooled variance is numerically zero.
 */
auto cohens_d(const std::vector<float>& a, const std::vector<float>& b) -> float;

/**
 * Two-sided Welch-style t-test p-value using normal CDF approximation.
 * Returns 1 when sample sizes are insufficient or numerical instability occurs.
 */
auto t_test_pvalue_approx(const std::vector<float>& a, const std::vector<float>& b) -> float;

/**
 * Two-sided Wilcoxon signed-rank p-value using normal CDF approximation.
 * Returns 1 when paired inputs are invalid or numerical instability occurs.
 */
auto wilcoxon_signed_rank_pvalue_approx(const std::vector<float>& a, const std::vector<float>& b)
    -> float;

} // namespace statistics

#endif // NN_STATISTICS_INFERENCE_TESTS_HPP