#ifndef NN_MODELS_GRU_GRUAUTOENCODERCONFIG_HPP
#define NN_MODELS_GRU_GRUAUTOENCODERCONFIG_HPP

/**
 * @file include/models/gru/GRUAutoencoderConfig.hpp
 * @brief Configuration struct for GRUAutoencoder (mirrors LSTMAutoencoderConfig).
 */

namespace nn::models::gru
{

/// Configuration controlling the GRU autoencoder dimensions.
struct GRUAutoencoderConfig
{
    int input_size = 64;   ///< D: number of input features per time step
    int seq_len = 32;      ///< T: expected sequence length (used for MAC estimation)
    int hidden_size = 128; ///< H: hidden dimension in each GRU layer
    int latent_size = 16;  ///< Z: bottleneck (latent) dimension
    int num_layers = 1;    ///< number of stacked GRU layers in encoder and decoder
};

} // namespace nn::models::gru

#endif // NN_MODELS_GRU_GRUAUTOENCODERCONFIG_HPP
