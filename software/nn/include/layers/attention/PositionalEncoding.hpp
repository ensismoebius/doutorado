#ifndef NN_LAYERS_ATTENTION_POSITIONALENCODING_HPP
#define NN_LAYERS_ATTENTION_POSITIONALENCODING_HPP

/**
 * @file include/layers/attention/PositionalEncoding.hpp
 * @brief Fixed sinusoidal positional encoding (Vaswani et al., 2017, §3.5).
 *
 *   PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
 *   PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
 *
 * It has no parameters and no learnable state: a model adds the returned (T,
 * d_model) matrix to its embeddings, and the gradient w.r.t. the embeddings
 * passes straight through (the addition's Jacobian is the identity). Hence this
 * is a free function rather than a Module.
 */

#include <cmath>

#include "tensor/Tensor.hpp"

namespace nn::layers
{

template <typename Backend>
inline auto sinusoidal_positional_encoding(int seq_len, int d_model) -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> pe(static_cast<nn::Index>(seq_len), static_cast<nn::Index>(d_model));
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int i = 0; i < d_model; ++i)
        {
            const int pair = i / 2;
            const double denom = std::pow(10000.0, (2.0 * pair) / static_cast<double>(d_model));
            const double angle = static_cast<double>(pos) / denom;
            pe.at(static_cast<nn::Index>(pos), static_cast<nn::Index>(i)) =
                static_cast<float>((i % 2 == 0) ? std::sin(angle) : std::cos(angle));
        }
    }
    return pe;
}

} // namespace nn::layers

#endif // NN_LAYERS_ATTENTION_POSITIONALENCODING_HPP
