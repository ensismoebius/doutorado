#pragma once

/**
 * @file LSTMAutoencoder.hpp
 * @brief LSTM-based autoencoder for 1-D temporal signals.
 *
 * Architecture:
 *
 *   Encoder:
 *     LSTM_enc (D → H_enc)  — runs over T time steps
 *     Final hidden state h_T : [1 × H_enc]
 *     Linear projection     : [1 × H_enc] → [1 × latent_size]
 *                             (followed by tanh for bounded latent codes)
 *
 *   Decoder:
 *     Linear expansion      : [1 × latent_size] → [1 × H_dec]
 *     Repeat h_0 for T steps (teacher-forcing not used; h is repeated)
 *     LSTM_dec (H_dec → H_dec) — produces [T × H_dec]
 *     Linear output proj    : [T × H_dec] → [T × D]
 *
 *   Loss: MSE between reconstruction [T × D] and original input [T × D].
 *
 * Sequence handling assumption:
 *   Each sample is a fixed-length 2-D tensor [T × D] where T is the number
 *   of time frames and D is the feature dimension per frame. This matches
 *   the windowed dataset convention used by experiment03.
 */

#include <memory>
#include <span>
#include <vector>

#include "LSTMLayer.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

namespace experiment04
{

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct LSTMAutoencoderConfig
{
    int input_size  = 64;    ///< D — feature dimension per time step
    int seq_len     = 32;    ///< T — number of time steps (fixed-length assumption)
    int hidden_size = 128;   ///< H — LSTM hidden dimension
    int latent_size = 16;    ///< Z — bottleneck dimension
    int num_layers  = 1;     ///< number of stacked LSTM layers (encoder and decoder each)
};

// ---------------------------------------------------------------------------
// LSTMAutoencoder
// ---------------------------------------------------------------------------
class LSTMAutoencoder : public Module<nn::EigenTensorBackend>
{
public:
    using Tensor = nn::Tensor;

    LSTMAutoencoderConfig cfg_;

    // --- Encoder stack ---
    std::vector<std::unique_ptr<LSTMLayer>> enc_lstms_;

    // Projection from last encoder hidden state → latent
    std::unique_ptr<Linear> enc_proj_;   // [hidden_size → latent_size]

    // --- Decoder stack ---
    // Expansion from latent → first decoder hidden dim
    std::unique_ptr<Linear> dec_expand_; // [latent_size → hidden_size]

    // Decoder LSTM layers  (input to first layer = hidden_size, output = hidden_size)
    std::vector<std::unique_ptr<LSTMLayer>> dec_lstms_;

    // Output projection: hidden_size → input_size  (applied to each time step)
    std::unique_ptr<Linear> out_proj_;   // [hidden_size → input_size]

    // --- Parameter flat view (for optimizer) ---
    std::vector<nn::Tensor*> param_ptrs_;

    // --- Forward caches (for backward) ---
    Tensor enc_output_cache_;    // [T × H]  — all encoder hidden states
    Tensor latent_cache_;        // [1 × Z]  — tanh output
    Tensor latent_pre_cache_;    // [1 × Z]  — before tanh (needed for grad)
    Tensor dec_input_cache_;     // [T × H]  — repeated latent expansion for decoder LSTM input
    Tensor dec_output_cache_;    // [T × H]  — decoder LSTM output
    Tensor recon_cache_;         // [T × D]  — reconstruction before loss
    bool requires_grad_ = false;

    explicit LSTMAutoencoder(const LSTMAutoencoderConfig& cfg);

    // ------- Forward / Backward -------

    /**
     * @brief Encode input sequence to latent vector.
     * @param input [T × D]
     * @return [1 × latent_size]
     */
    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;

    /**
     * @brief Decode latent vector to reconstructed sequence.
     * @param latent [1 × latent_size]
     * @return [T × D]
     */
    auto decode(const Tensor& latent, int seq_len, bool requires_grad = true) -> Tensor;

    /**
     * @brief Full forward: encode then decode. Returns [T × D] reconstruction.
     */
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;

    /**
     * @brief Backward pass through the full autoencoder.
     * @param grad_output Gradient of loss w.r.t. reconstruction [T × D].
     * @return Gradient w.r.t. input sequence [T × D].
     */
    auto backward(const Tensor& grad_output) -> Tensor override;

    void reset_state() override;

    auto params() -> std::span<nn::Tensor*> override;

    auto state_dict() const -> std::map<std::string, nn::Tensor> override;
    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override;

private:
    void build_param_ptrs();
};

} // namespace experiment04
