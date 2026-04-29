#ifndef NN_MODELS_LSTM_LSTMAUTOENCODER_HPP
#define NN_MODELS_LSTM_LSTMAUTOENCODER_HPP

/**
 * @file include/nn/models/lstm/LSTMAutoencoder.hpp
 * @brief LSTM-based autoencoder for 1-D temporal signals.
 *
 * Architecture
 * ------------
 * Encoder: stacked LSTMLayer → projection to latent (tanh)
 * Decoder: latent expand → replicate T times → stacked LSTMLayer → output projection
 *
 * The model owns hidden state and must be called with @c reset_state() between
 * independent sequences.  Both @c encode and @c decode accept a @c requires_grad
 * flag that activates caching for BPTT.
 */

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/models/lstm/LSTMLayer.hpp"
#include "nn/tensor/Tensor.hpp"

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

/**
 * @class LSTMAutoencoder
 * @brief Sequence-to-sequence LSTM autoencoder with full BPTT.
 *
 * Forward pass (sequence of length T, dimensionality D):
 *   1. Encoder: T×D → stacked LSTM → last hidden → Linear → tanh → 1×Z latent
 *   2. Decoder: 1×Z → Linear expand → replicate to T×H → stacked LSTM → Linear → T×D recon
 *
 * Backward pass uses the caches filled during forward to propagate gradients
 * through all LSTM steps (BPTT) and the encoder/decoder projections.
 */
class LSTMAutoencoder : public Module<nn::Backend>
{
   public:
    using Tensor = nn::Tensor;

    LSTMAutoencoderConfig cfg_; ///< architecture configuration (read-only after construction)

    // Encoder components
    std::vector<std::unique_ptr<LSTMLayer>> enc_lstms_; ///< stacked encoder LSTM layers
    std::unique_ptr<Linear> enc_proj_;                  ///< H → Z linear projection

    // Decoder components
    std::unique_ptr<Linear> dec_expand_;                ///< Z → H expand projection
    std::vector<std::unique_ptr<LSTMLayer>> dec_lstms_; ///< stacked decoder LSTM layers
    std::unique_ptr<Linear> out_proj_;                  ///< H → D output projection

    std::vector<nn::Tensor*> param_ptrs_; ///< flat parameter list for optimizer

    // BPTT caches (populated during forward when requires_grad is true)
    Tensor enc_output_cache_; ///< full encoder hidden output (T × H)
    Tensor latent_cache_;     ///< post-tanh latent (1 × Z)
    Tensor latent_pre_cache_; ///< pre-tanh latent (1 × Z)
    Tensor dec_input_cache_;  ///< expanded + replicated decoder input (T × H)
    Tensor dec_output_cache_; ///< full decoder hidden output (T × H)
    Tensor recon_cache_;      ///< reconstruction (T × D)
    bool requires_grad_ = false;

    /// Construct and initialise all weights with N(0, 0.05).
    explicit LSTMAutoencoder(const LSTMAutoencoderConfig& cfg);

    /// Encode an input sequence to a latent vector.
    /// @param input         (T × D) input
    /// @param requires_grad enable BPTT caching
    /// @return (1 × Z) latent
    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;

    /// Decode a latent vector back to a sequence.
    /// @param latent        (1 × Z) latent
    /// @param seq_len       target output length T
    /// @param requires_grad enable BPTT caching
    /// @return (T × D) reconstruction
    auto decode(const Tensor& latent, int seq_len, bool requires_grad = true) -> Tensor;

    /// Full forward pass: encode then decode.
    /// @param input         (T × D) input sequence
    /// @param requires_grad enable BPTT caching
    /// @return (T × D) reconstruction
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;

    /// BPTT backward pass; must be called after forward with requires_grad=true.
    /// @param grad_output (T × D) gradient of the loss w.r.t. the reconstruction
    /// @return (T × D) gradient w.r.t. the input
    auto backward(const Tensor& grad_output) -> Tensor override;

    /// Reset all LSTM hidden and cell states to zero.
    void reset_state() override;

    /// Flat parameter pointer list for use with optimizers.
    auto params() -> std::span<nn::Tensor*> override;

    /// Serialise all parameters to a named map.
    auto state_dict() const -> std::map<std::string, nn::Tensor> override;

    /// Load parameters from a named map (e.g. produced by state_dict()).
    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override;

   private:
    /// Populate param_ptrs_ from all sub-modules.
    void build_param_ptrs();
};

} // namespace nn::models::lstm

#endif // NN_MODELS_LSTM_LSTMAUTOENCODER_HPP
