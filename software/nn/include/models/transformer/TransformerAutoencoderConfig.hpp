#ifndef NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODERCONFIG_HPP
#define NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODERCONFIG_HPP

/**
 * @file include/models/transformer/TransformerAutoencoderConfig.hpp
 * @brief Configuration for the bottlenecked Transformer autoencoder.
 */

#include <cstdint>
#include <optional>

namespace nn::models::transformer
{

struct TransformerAutoencoderConfig
{
    int input_size = 8;   ///< D: features per time step (the framed window)
    int seq_len = 32;     ///< T: sequence length
    int d_model = 64;     ///< model / embedding dimension
    int n_heads = 4;      ///< attention heads (must divide d_model)
    int n_layers = 2;     ///< encoder blocks in the encoder AND in the decoder
    int d_ff = 128;       ///< position-wise FFN hidden size
    int latent_size = 32; ///< Z: bottleneck dimension
    /// Pre-norm blocks (Xiong et al., ICML 2020) and a final LayerNorm closing each stack,
    /// instead of the post-norm blocks of Vaswani et al. 2017 (TransformerEncoderBlock.hpp).
    bool norm_first = false;
    /// Unset (the default): every weight from a seed fixed in the model's code, so two runs
    /// start from the same weights whatever their seed. Set: every weight tensor from
    /// nn::utility::derive_seed(*init_seed, <its stream>), distinct per layer and tensor.
    std::optional<std::uint32_t> init_seed;
};

} // namespace nn::models::transformer

#endif // NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODERCONFIG_HPP
