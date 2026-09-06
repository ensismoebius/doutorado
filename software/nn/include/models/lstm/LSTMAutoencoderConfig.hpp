#ifndef NN_MODELS_LSTM_LSTMAUTOENCODERCONFIG_HPP
#define NN_MODELS_LSTM_LSTMAUTOENCODERCONFIG_HPP

/**
 * @file include/models/lstm/LSTMAutoencoderConfig.hpp
 * @brief Configuration struct for LSTMAutoencoder (extracted from LSTMAutoencoder.hpp).
 */

namespace nn::models::lstm
{

/// Configuration controlling the LSTM autoencoder dimensions.
struct LSTMAutoencoderConfig
{
    int input_size = 64;   ///< D: number of input features per time step
    int seq_len = 32;      ///< T: expected sequence length (used for MAC estimation)
    int hidden_size = 128; ///< H: hidden dimension in each LSTM layer
    int latent_size = 16;  ///< Z: bottleneck (latent) dimension
    int num_layers = 1;    ///< number of stacked LSTM layers in encoder and decoder
};

} // namespace nn::models::lstm

#endif // NN_MODELS_LSTM_LSTMAUTOENCODERCONFIG_HPP
