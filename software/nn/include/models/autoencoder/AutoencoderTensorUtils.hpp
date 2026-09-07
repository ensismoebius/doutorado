/**
 * @file include/models/autoencoder/AutoencoderTensorUtils.hpp
 * @brief Small tensor/param plumbing utilities (extracted from AutoencoderBuilders.hpp).
 */

#ifndef NN_MODELS_AUTOENCODER_AUTOENCODER_TENSOR_UTILS_HPP
#define NN_MODELS_AUTOENCODER_AUTOENCODER_TENSOR_UTILS_HPP

#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "layers/Layers.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

using Tensor = nn::Tensor;

inline auto slice_columns(const Tensor& input, nn::Index col_offset, nn::Index col_count) -> Tensor
{
    return input.block(0, col_offset, input.rows(), col_count);
}

inline auto concat_columns(const Tensor& left, const Tensor& right) -> Tensor
{
    if (left.rows() != right.rows())
        throw std::invalid_argument("concat_columns requires equal row counts");

    Tensor joined(left.rows(), left.cols() + right.cols());
    joined.setBlock(0, 0, left);
    joined.setBlock(0, left.cols(), right);
    return joined;
}

inline auto join_params(std::initializer_list<std::vector<Tensor*>> groups) -> std::vector<Tensor*>
{
    std::vector<Tensor*> params;
    for (const auto& group : groups) params.insert(params.end(), group.begin(), group.end());
    return params;
}

inline void reset_sequential_state(Sequential& seq)
{
    for (auto& layer : seq.layers)
    {
        layer->reset_state();
    }
}

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUTOENCODER_TENSOR_UTILS_HPP
