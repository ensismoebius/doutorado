#ifndef NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODERCONFIG_HPP
#define NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODERCONFIG_HPP

/**
 * @file include/models/transformer/TransformerAutoencoderConfig.hpp
 * @brief Configuration for the bottlenecked Transformer autoencoder.
 */

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
};

} // namespace nn::models::transformer

#endif // NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODERCONFIG_HPP
