#ifndef NN_LAYERS_CONV2D_PATCHINDICES_HPP
#define NN_LAYERS_CONV2D_PATCHINDICES_HPP

#include <utility>
#include <vector>

/**
 * @file PatchIndices.hpp
 * @brief Cache structure for precomputed patch indices used in im2col/col2im
 *        operations (extracted from Conv2d_utils.hpp).
 */

namespace Conv2dUtils
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

} // namespace Conv2dUtils

#endif // NN_LAYERS_CONV2D_PATCHINDICES_HPP
