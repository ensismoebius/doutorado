#ifndef NN_UTILITY_SIGNAL_PREPROCESSING_HPP
#define NN_UTILITY_SIGNAL_PREPROCESSING_HPP

#include <filesystem>

#include "tensor/Tensor.hpp"

namespace nn::utility
{

/**
 * @brief Read numeric values from a CSV/TXT signal file into a column tensor.
 *
 * Non-numeric tokens are ignored so mixed-content rows can still be consumed.
 * The returned tensor has shape `(N, 1)`, where `N` is the count of parsed
 * numeric tokens.
 */
auto read_csv_signal(const std::filesystem::path& path) -> nn::Tensor;

/**
 * @brief In-place z-score normalization over all tensor elements.
 *
 * Computes mean and population standard deviation over the entire tensor and
 * rewrites each element as `(x - mean) / stddev`. For near-constant inputs, an
 * epsilon floor is applied to the variance for numerical stability.
 */
void zscore_inplace(nn::Tensor& signal);

} // namespace nn::utility

#endif // NN_UTILITY_SIGNAL_PREPROCESSING_HPP
