#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../tensor/Tensor.hpp"

// Forward declaration
class Conv2d;

/**
 * @file Conv2d_utils.hpp
 * @brief Helper structures and private utilities for Conv2d convolution layer.
 *
 * This header contains internal implementation details for the Conv2d class,
 * including index caching and helper function declarations.
 */

namespace Conv2dImpl
{
/**
 * @struct PatchIndices
 * @brief Cache structure for precomputed patch indices used in im2col/col2im operations.
 */
struct PatchIndices
{
    std::vector<float> values;                  ///< Values for im2col operation
    std::vector<std::pair<int, int>> positions; ///< (row, col) positions for col2im
};

/**
 * @brief Hash function for std::pair<int, int> for use in unordered_map.
 */
struct PairHash
{
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U>& p) const
    {
        return std::hash<T>()(p.first) ^ (std::hash<U>()(p.second) << 1);
    }
};

using IndexCache = std::unordered_map<std::pair<int, int>, std::vector<PatchIndices>, PairHash>;

} // namespace Conv2dImpl
