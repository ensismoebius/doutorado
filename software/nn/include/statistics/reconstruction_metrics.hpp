#ifndef SRC_CORE_STATISTICS_RECONSTRUCTIONMETRICS_H_
#define SRC_CORE_STATISTICS_RECONSTRUCTIONMETRICS_H_

#include <cmath>
#include <stdexcept>

namespace statistics
{

/**
 * @file reconstruction_metrics.hpp
 * @brief Elementwise reconstruction error between two equal-shaped tensors.
 *
 * Plain scalar metrics for reporting/comparison (autoencoder reconstruction quality,
 * regression error), not trainable loss modules -- for a loss with a backward pass, see
 * layers/losses/MSELoss.hpp / MAELoss.hpp instead.
 */

template <typename Tensor>
auto mse_between(const Tensor& a, const Tensor& b) -> float
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("mse_between shape mismatch");
    }
    float sum = 0.0f;
    const auto n = a.size();
    for (std::remove_const_t<decltype(n)> i = 0; i < n; ++i)
    {
        const float d = a.at(i) - b.at(i);
        sum += d * d;
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : 0.0f;
}

template <typename Tensor>
auto mae_between(const Tensor& a, const Tensor& b) -> float
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("mae_between shape mismatch");
    }
    float sum = 0.0f;
    const auto n = a.size();
    for (std::remove_const_t<decltype(n)> i = 0; i < n; ++i)
    {
        sum += std::fabs(a.at(i) - b.at(i));
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : 0.0f;
}

} // namespace statistics

#endif /* SRC_CORE_STATISTICS_RECONSTRUCTIONMETRICS_H_ */
