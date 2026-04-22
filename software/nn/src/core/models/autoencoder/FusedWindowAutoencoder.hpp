/**
 * @file FusedWindowAutoencoder.hpp
 * @brief Dual-branch autoencoder for multimodal (EEG + Audio) data.
 *
 * This is a multimodal autoencoder that processes EEG and Audio signals
 * through SEPARATE encoders, then fuses them before decoding.
 *
 * Architecture Pattern (Dual-Branch Fusion):
 *   EEG Input ----> [EEG Encoder] ----\
 *                                       |---> [Fusion] --> [Decoder] --> Reconstruction
 *   Audio Input --> [Audio Encoder] --/
 *
 * This design allows:
 *   1. Modality-specific feature extraction
 *   2. Learning shared representations
 *   3. Handling missing modalities (can run with only one branch)
 *
 * Use Cases:
 *   - Imagined speech with both EEG and audio (10.1117 protocol)
 *   - Emotion recognition from facial expression + physiological signals
 *   - Any domain with complementary sensor modalities
 *
 * @note For fused spiking variant, see FusedWindowSpikingAutoencoder
 */
#ifndef NN_MODELS_AUTOENCODER_FUSED_WINDOW_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_FUSED_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "BaseAutoencoder.hpp"
#include "Config.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

/**
 * @class FusedWindowAutoencoder
 * @brief Dual-branch autoencoder for multimodal (EEG + Audio) data.
 *
 * This model learns joint representations from two input modalities.
 * The key innovation is the fusion layer that combines both encoded streams.
 *
 * Fusion Strategy:
 *   1. Encode EEG: eeg_features -> hidden_eeg -> latent_eeg
 *   2. Encode Audio: audio_features -> hidden_audio -> latent_audio
 *   3. Fuse: concat(latent_eeg, latent_audio) -> fusion_hidden -> latent_shared
 *   4. Decode: latent_shared -> hidden -> audio_features
 *
 * @note You MUST set eeg_features and audio_features in config
 *       before constructing this model.
 */
class FusedWindowAutoencoder : public BaseAutoencoder<nn::EigenTensorBackend>
{
   public:
    /**
     * @brief Construct fused multimodal autoencoder
     *
     * @param cfg Configuration with BOTH eeg_features and audio_features set
     *
     * @throws std::invalid_argument if eeg_features or audio_features is 0
     */
    explicit FusedWindowAutoencoder(const AutoencoderConfig& cfg);

    /**
     * @brief Encode multimodal input to fused latent representation
     *
     * @param eeg_input EEG features (batch_size x eeg_features)
     * @param audio_input Audio features (batch_size x audio_features)
     * @param requires_grad Whether to track gradients
     * @return Fused latent (batch_size x latent_size)
     *
     * @note Alternative API: pass single tensor with concatenated [eeg|audio]
     */
    auto encode_bimodal(
        const nn::Tensor& eeg_input, const nn::Tensor& audio_input, bool requires_grad = true)
        -> nn::Tensor;

    /**
     * @brief Encode single input (monomodal fallback)
     */
    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;

    /**
     * @brief Decode from fused latent to both modalities
     *
     * @param latent Shared latent (batch_size x latent_size)
     * @param requires_grad Whether to track gradients
     * @return Reconstruction (batch_size x eeg_features + audio_features)
     */
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor override;

    /**
     * @brief Get architecture name
     * @return "FusedWindowAutoencoder"
     */
    auto architecture_name() const -> std::string override
    {
        return "FusedWindowAutoencoder";
    }

    /**
     * @brief Return all trainable parameters
     */
    auto params() -> std::span<nn::Tensor*> override;

    /**
     * @brief Get encoded EEG features separately
     */
    auto encode_eeg_only(const nn::Tensor& eeg_input, bool requires_grad = true) -> nn::Tensor;

    /**
     * @brief Get encoded Audio features separately
     */
    auto encode_audio_only(const nn::Tensor& audio_input, bool requires_grad = true) -> nn::Tensor;

   private:
    void build_bimodal_encoder(const AutoencoderConfig& cfg);
    void build_decoder(const AutoencoderConfig& cfg);

    // Branch encoders (like PyTorch's branch1, branch2)
    Sequential eeg_encoder_;
    Sequential audio_encoder_;

    // Fusion layer (combines both branches)
    Sequential fusion_;

    // Shared decoder
    Sequential decoder_;

    std::vector<nn::Tensor*> param_ptrs_;

    int eeg_features_ = 0;
    int audio_features_ = 0;
};

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_FUSED_WINDOW_AUTOENCODER_HPP
