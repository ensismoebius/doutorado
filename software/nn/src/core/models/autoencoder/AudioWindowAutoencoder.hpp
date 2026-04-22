/**
 * @file AudioWindowAutoencoder.hpp
 * @brief ANN autoencoder for audio window features.
 *
 * This class implements a symmetric MLP autoencoder for audio data,
 * following the PyTorch pattern of nn.Module.
 *
 * Architecture (Symmetric Design):
 *   Encoder: Input -> Linear(in→h) -> ReLU -> ... -> Linear(h→latent) -> ReLU
 *   Decoder: Latent -> Linear(latent→h) -> ReLU -> ... -> Linear(h→out) -> ReLU
 *
 * The name "Window" refers to time-windowed audio features (e.g., MFCCs, filterbanks).
 * Typical input: 128-dimensional feature vector per time window.
 *
 * Design Pattern (PyTorch-Style):
 *   1. Inherit from BaseAutoencoder<EigenTensorBackend>
 *   2. Use Sequential for encoder/decoder (like torch.nn.Sequential)
 *   3. Override encode() and decode() for custom logic
 *   4. Implement params() to return all trainable tensors
 *
 * Example Usage:
 *   @code
 *   AutoencoderConfig cfg{
 *       .input_features = 128,   // e.g., 64 mel bands
 *       .hidden_size = 64,
 *       .latent_size = 32,
 *       .depth = 2
 *   };
 *
 *   AudioWindowAutoencoder model(cfg);
 *
 *   // Forward pass (encode + decode)
 *   nn::Tensor input = nn::Tensor::randn({16, 128});  // batch=16
 *   nn::Tensor reconstruction = model.forward(input);
 *
 *   // Or just encode (for feature extraction)
 *   nn::Tensor latent = model.encode(input);
 *   @endcode
 *
 * @note This is an ANN (Artificial Neural Network) model using ReLU.
 *       For spiking (SNN) version, see AudioWindowSpikingAutoencoder.
 */
#ifndef NN_MODELS_AUTOENCODER_AUDIO_WINDOW_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_AUDIO_WINDOW_AUTOENCODER_HPP

#include <vector>

#include "BaseAutoencoder.hpp"
#include "Config.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @namespace nn::models::autoencoder
 * @brief Core autoencoder model implementations
 */
namespace nn::models::autoencoder
{

/**
 * @class AudioWindowAutoencoder
 * @brief Standard ANN autoencoder for audio window features.
 *
 * This model learns to compress and reconstruct audio feature vectors.
 * The symmetric architecture helps ensure the latent space captures
 * meaningful audio representations.
 *
 * Layer Configuration:
 *   - depth=1: Input -> [Linear(128→64) -> ReLU -> Linear(64→32) -> ReLU]
 *                      [Linear(32→64) -> ReLU -> Linear(64→128) -> ReLU]
 *   - depth=2: Adds additional hidden layers
 *
 * @note Uses ReLU activation (standard ANN). For spiking neurons,
 *       use AudioWindowSpikingAutoencoder instead.
 */
class AudioWindowAutoencoder : public BaseAutoencoder<nn::EigenTensorBackend>
{
   public:
    /**
     * @brief Construct autoencoder from configuration
     *
     * @param cfg Configuration object with architecture hyperparameters
     *
     * The constructor builds:
     *   1. encoder_: Sequential network shrinking input→latent
     *   2. decoder_: Sequential network expanding latent→input
     *
     * Design: Uses depth to create symmetric hidden layers.
     *         Each depth level adds: Linear(h→h/2) on encoder, Linear(h/2→h) on decoder
     */
    explicit AudioWindowAutoencoder(const AutoencoderConfig& cfg);

    /**
     * @brief Encode audio features to latent representation
     *
     * Compresses the input feature vector through the encoder network.
     * The output is a lower-dimensional representation that captures
     * the essential characteristics of the audio window.
     *
     * @param input Audio features (batch_size x input_features)
     * @param requires_grad Whether to track gradients
     * @return Latent representation (batch_size x latent_size)
     */
    auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override;

    /**
     * @brief Decode latent representation to audio reconstruction
     *
     * Reconstructs the original audio feature dimension from the
     * compressed latent representation.
     *
     * @param latent Compressed representation (batch_size x latent_size)
     * @param requires_grad Whether to track gradients
     * @return Reconstruction (batch_size x input_features)
     */
    auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor override;

    /**
     * @brief Get architecture name for logging
     * @return "AudioWindowAutoencoder"
     */
    auto architecture_name() const -> std::string override
    {
        return "AudioWindowAutoencoder";
    }

    /**
     * @brief Return all trainable parameters
     *
     * Required by nn::Module for optimizer attachment.
     * Returns concatenated encoder + decoder parameter pointers.
     *
     * @return Span of pointers to parameter tensors
     */
    auto params() -> std::span<nn::Tensor*> override;

   private:
    /**
     * @brief Build encoder network from config
     *
     * Creates Sequential layers:
     *   Linear(input→hidden) -> ReLU
     *   ... (depth times) ...
     *   Linear(hidden→latent) -> ReLU
     *
     * @param cfg Configuration with architecture details
     */
    void build_encoder(const AutoencoderConfig& cfg);

    /**
     * @brief Build decoder network from config
     *
     * Creates Sequential layers (mirror of encoder):
     *   Linear(latent→hidden) -> ReLU
     *   ... (depth times) ...
     *   Linear(hidden→input) -> ReLU
     *
     * @param cfg Configuration with architecture details
     */
    void build_decoder(const AutoencoderConfig& cfg);

    // ==================== PyTorch-style Model Components ====================
    /** @brief Encoder network (input → latent) - like torch.nn.Encoder */
    Sequential encoder_;

    /** @brief Decoder network (latent → input) - like torch.nn.Decoder */
    Sequential decoder_;

    /** @brief Cached parameter pointers for optimizer attachment */
    std::vector<nn::Tensor*> param_ptrs_;
};

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUDIO_WINDOW_AUTOENCODER_HPP
