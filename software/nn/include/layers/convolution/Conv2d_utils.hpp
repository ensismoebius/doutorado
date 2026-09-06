#ifndef NN_LAYERS_CONV2D_UTILS_HPP
#define NN_LAYERS_CONV2D_UTILS_HPP

#include <unordered_map>
#include <vector>

#include "layers/convolution/PatchIndices.hpp"

// Forward declaration
template <typename Backend>
class Conv2dImpl;

/**
 * @file Conv2d_utils.hpp
 * @brief Helper structures and private utilities for Conv2d convolution layer.
 *
 * This header contains internal implementation details for the Conv2d class,
 * including index caching and helper function declarations.
 *
 * Why caching exists:
 * - The im2col/col2im implementation needs to know, for each output spatial
 *   position, which input elements participate in the convolution patch.
 * - Those index patterns depend on (batch_size, input_height, input_width) and
 *   layer hyperparameters (kernel/stride/padding/dilation).
 * - Precomputing them once and reusing them avoids repeated integer math in
 *   hot loops, at the cost of memory.
 *
 * `PatchIndices` itself lives in PatchIndices.hpp (it is used by Conv2d.hpp
 * directly); this header stays as a thin include for existing includers plus
 * the local-only `TripleHash` helper and `IndexCache` alias.
 */

namespace Conv2dUtils
{
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

// Implementation note:
// - The cache key is a (batch_size, input_height, input_width) tuple.
// - The cached value contains the patch extraction mapping for those dimensions.
// - Thread safety is handled on the Conv2d side (mutex around cache updates).

} // namespace Conv2dUtils

#endif // NN_LAYERS_CONV2D_UTILS_HPP
