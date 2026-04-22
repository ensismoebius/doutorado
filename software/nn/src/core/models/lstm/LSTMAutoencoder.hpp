/**
 * @file LSTMAutoencoder.hpp
 * @brief LSTM-based sequence-to-sequence autoencoder.
 *
 * This model uses Long Short-Term Memory (LSTM) layers for sequential data,
 * following the PyTorch pattern for RNN-based autoencoders.
 *
 * LSTM Theory:
 *   - LSTM cells have: input gate, forget gate, output gate, cell state
 *   - Cell state: long-term memory that can persist across many timesteps
 *   - Gates: control information flow (what to remember, what to forget)
 *
 * Architecture:
 *   Encoder: LSTM(input) -> (hidden_state, cell_state)
 *   Decoder: LSTM(latent + initial_cell) -> sequence
 *
 * Use Cases:
 *   - Time-series compression (audio waveforms, EEG epochs)
 *   - Sequence-to-sequence tasks (machine translation, etc.)
 *   - Anomaly detection in temporal data
 *
 * @note This follows PyTorch's nn.LSTM pattern but wrapped as an autoencoder.
 */
#ifndef NN_MODELS_LSTM_LSTMAUTOENCODER_HPP
#define NN_MODELS_LSTM_LSTMAUTOENCODER_HPP

#include <vector>
#include <span>

#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::models::lstm
{

/**
 * @struct LSTMAutoencoderConfig
 * @brief Configuration for LSTM autoencoder.
 */
struct LSTMAutoencoderConfig
{
    /** @brief Input feature dimension */
    int input_size = 128;

    /** @brief Hidden layer dimension */
    int hidden_size = 64;

    /** @brief Latent representation dimension */
    int latent_size = 32;

    /** @brief Number of LSTM layers */
    int num_layers = 1;

    /** @brief Dropout probability (0-1) */
    float dropout = 0.0F;

    /** @brief Whether to use bidirectional LSTM */
    bool bidirectional = false;
};

/**
 * @class LSTMAutoencoder
 * @brief Sequence autoencoder using LSTM layers.
 *
 * This model encodes a sequence into a fixed-size latent representation,
 * then decodes that latent back to a sequence.
 *
 * PyTorch equivalent:
 *   encoder = nn.LSTM(input_size, hidden_size, num_layers)
 *   decoder = nn.LSTM(latent_size, hidden_size, num_layers)
 *
 * Usage:
 *   @code
 *   LSTMAutoencoderConfig cfg{
 *       .input_size = 128,
 *       .hidden_size = 64,
 *       .latent_size = 32,
 *       .num_layers = 2
 *   };
 *   LSTMAutoencoder model(cfg);
 *
 *   // Encode sequence
 *   nn::Tensor input = nn::Tensor::randn({seq_len, batch, input_size});
 *   auto [latent, state] = model.encode_sequence(input);
 *
 *   // Decode sequence
 *   auto output = model.decode_sequence(latent, state, target_length);
 *   @endcode
 */
class LSTMAutoencoder : public nn::Module<nn::EigenTensorBackend>
{
public:
    /**
     * @brief Construct LSTM autoencoder
     * @param cfg Configuration with architecture details
     */
    explicit LSTMAutoencoder(const LSTMAutoencoderConfig& cfg);

    /**
     * @brief Encode sequence to latent representation
     *
     * @param input Sequence tensor (timesteps x batch x input_size)
     * @return Tuple of (last_hidden, final_cell_state)
     */
    auto encode_sequence(const nn::Tensor& input)
        -> std::pair<nn::Tensor, nn::Tensor>;

    /**
     * @brief Decode latent to sequence
     *
     * @param latent Initial hidden state
     * @param cell Initial cell state
     * @param target_length Number of output timesteps
     * @return Reconstructed sequence (target_length x batch x input_size)
     */
    auto decode_sequence(
        const nn::Tensor& latent,
        const nn::Tensor& cell,
        int target_length
    ) -> nn::Tensor;

    /**
     * @brief Forward pass (encode -> decode)
     * @param input Input sequence
     * @return Reconstructed sequence
     */
    auto forward(const nn::Tensor& input) -> nn::Tensor override;

    /**
     * @brief Reset hidden state for new sequence
     */
    void reset_state();

    /**
     * @brief Get all trainable parameters
     * @return Span of parameter pointers
     */
    auto params() -> std::span<nn::Tensor*> override;

    /**
     * @brief Get architecture name
     * @return "LSTMAutoencoder"
     */
    auto architecture_name() const -> std::string
    {
        return "LSTMAutoencoder";
    }

private:
    LSTMAutoencoderConfig config_;

    // PyTorch-style: encoder and decoder as separate modules
    nn::Sequential encoder_;
    nn::Sequential decoder_;

    std::vector<nn::Tensor*> param_ptrs_;

    // Hidden states (reset between sequences)
    nn::Tensor hidden_state_;
    nn::Tensor cell_state_;
};

} // namespace nn::models::lstm

#endif // NN_MODELS_LSTM_LSTMAUTOENCODER_HPP