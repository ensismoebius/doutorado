#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "nn/tensor/Tensor.hpp"

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
struct TripleHash
{
    template <typename T, typename U, typename V>
    std::size_t operator()(const std::tuple<T, U, V>& t) const
    {
        std::size_t h1 = std::hash<T>()(std::get<0>(t));
        std::size_t h2 = std::hash<U>()(std::get<1>(t));
        std::size_t h3 = std::hash<V>()(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

using IndexCache =
    std::unordered_map<std::tuple<int, int, int>, std::vector<PatchIndices>, TripleHash>;

} // namespace Conv2dImpl
