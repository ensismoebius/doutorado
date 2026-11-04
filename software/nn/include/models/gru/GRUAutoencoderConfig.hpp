#ifndef NN_MODELS_GRU_GRUAUTOENCODERCONFIG_HPP
#define NN_MODELS_GRU_GRUAUTOENCODERCONFIG_HPP

/**
 * @file include/models/gru/GRUAutoencoderConfig.hpp
 * @brief Configuration struct for GRUAutoencoder (mirrors LSTMAutoencoderConfig).
 */

#include <cstdint>
#include <optional>

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
    /// Unset (the default): every weight from a seed fixed in the model's code, so two runs
    /// start from the same weights whatever their seed. Set: every weight tensor from
    /// nn::utility::derive_seed(*init_seed, <its stream>), distinct per layer and tensor.
    std::optional<std::uint32_t> init_seed;
};

} // namespace nn::models::gru

#endif // NN_MODELS_GRU_GRUAUTOENCODERCONFIG_HPP
