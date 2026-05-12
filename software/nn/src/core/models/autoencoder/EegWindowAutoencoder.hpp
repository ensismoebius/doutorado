/**
 * @file EegWindowAutoencoder.hpp
 * @brief ANN autoencoder for EEG window features.
 *
 * Similar to AudioWindowAutoencoder but optimized for EEG (electroencephalography)
 * signal dimensions. EEG typically has different feature dimensions than audio.
 *
 * Architecture:
 *   Same symmetric MLP as AudioWindowAutoencoder, but expects different input size.
 *
 * Use Cases:
 *   - EEG-only imagined speech classification
 *   - Brain-computer interface feature extraction
 *   - EEG-based authentication/verification
 *
 * @note For spiking (SNN) EEG autoencoder, see EegWindowSpikingAutoencoder
 */
#ifndef NN_MODELS_AUTOENCODER_EEG_WINDOW_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_EEG_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "BaseAutoencoder.hpp"
#include "Config.hpp"
#include "layers/base/Module.hpp"
#include "layers/Layers.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

/**
 * @class EegWindowAutoencoder
 * @brief ANN autoencoder for EEG window features.
 *
 * Electroencephalography (EEG) signals typically have fewer channels than
 * audio features, so this model is configured for smaller input dimensions.
 *
 * Typical EEG Configuration:
 *   - input_features: 64 (from 64-channel EEG)
 *   - hidden_size: 32
 *   - latent_size: 16
 *
 * @note Uses ReLU activation (standard ANN). For spiking neurons,
 *       use EegWindowSpikingAutoencoder instead.
 */
class EegWindowAutoencoder : public BaseAutoencoder<nn::Backend>
{
   public:
    /**
     * @brief Construct EEG autoencoder from configuration
     * @param cfg Configuration with architecture hyperparameters
     */
    explicit EegWindowAutoencoder(const AutoencoderConfig& cfg);

    /**
     * @brief Encode EEG features to latent representation
     * @param input EEG features (batch_size x input_features)
     * @param requires_grad Whether to track gradients
     * @return Latent representation (batch_size x latent_size)
     */
    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;

    /**
     * @brief Decode latent representation to EEG reconstruction
     * @param latent Compressed representation
     * @param requires_grad Whether to track gradients
     * @return Reconstruction
     */
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor override;

    /**
     * @brief Get architecture name
     * @return "EegWindowAutoencoder"
     */
    auto architecture_name() const -> std::string override
    {
        return "EegWindowAutoencoder";
    }

    /**
     * @brief Return all trainable parameters
     */
    auto params() -> std::span<nn::Tensor*> override;

   private:
    void build_encoder(const AutoencoderConfig& cfg);
    void build_decoder(const AutoencoderConfig& cfg);

    Sequential encoder_;
    Sequential decoder_;
    std::vector<nn::Tensor*> param_ptrs_;
};

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_EEG_WINDOW_AUTOENCODER_HPP
