#ifndef NN_MODELS_AUTOENCODER_PROTOCOL_SPIKING_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_PROTOCOL_SPIKING_AUTOENCODER_HPP

#include <map>
#include <string>
#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

/**
 * @file ProtocolSpikingAutoencoder.hpp
 * @brief Spiking Neural Network (SNN) autoencoder for Protocol101117 features.
 *
 * Architecture:
 *   Encoder: Linear(input → hidden) → Lif → [×depth] → Linear(hidden → latent) → Lif
 *   Decoder: Linear(latent → hidden) → LifIntegrator → [×depth] → Linear(hidden → input)
 *
 * The encoder emits binary spike tensors; the decoder integrates spikes back to
 * continuous membrane potentials.  Call `reset_state()` between independent
 * sequences/trials to zero all membrane potentials.
 */
struct ProtocolSpikingAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    bool use_dual_branch_ = false;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    nn::Sequential eeg_encoder_;
    nn::Sequential audio_encoder_;
    nn::Sequential fusion_encoder_;
    nn::Sequential fusion_decoder_;
    nn::Sequential eeg_decoder_;
    nn::Sequential audio_decoder_;

    int eeg_features_ = 0;
    int audio_features_ = 0;

    // Firing-rate regularization (see AutoencoderConfig::firing_rate_reg_lambda).
    // Applies only to encoder Lif layers, identified once at construction time.
    float fr_lambda_ = 0.0F;
    float fr_min_ = 0.05F;
    float fr_max_ = 0.80F;
    std::vector<size_t> encoder_lif_indices_;
    std::vector<size_t> eeg_encoder_lif_indices_;
    std::vector<size_t> audio_encoder_lif_indices_;
    std::vector<size_t> fusion_encoder_lif_indices_;

    explicit ProtocolSpikingAutoencoder(const AutoencoderConfig& cfg);

    /// Run the encoder and return spike tensors for the latent layer.
    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;

    /// Run the decoder and return reconstructed continuous activations.
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;

    /// Reset all stateful (membrane potential) layers between sequences.
    void reset_state() override;

    /// Merge every sub-Sequential's state_dict, prefixed by its member name
    /// ("encoder.", "decoder.", "eeg_encoder.", ...). Whichever sub-Sequentials this
    /// profile did not build (e.g. eeg_encoder_ when use_dual_branch_ is false) are
    /// empty and contribute nothing — no modality-specific branching needed here.
    auto state_dict() const -> std::map<std::string, Tensor> override;
    void load_state_dict(const std::map<std::string, Tensor>& sd) override;
};
} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_PROTOCOL_SPIKING_AUTOENCODER_HPP
