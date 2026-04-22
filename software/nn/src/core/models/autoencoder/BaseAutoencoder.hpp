/**
 * @file BaseAutoencoder.hpp
 * @brief Abstract base class for all autoencoder models.
 *
 * This class defines the common interface for all autoencoder implementations
 * following the PyTorch/SNNTorch pattern:
 *   - encode(): compress input to latent representation
 *   - decode(): reconstruct from latent representation
 *   - forward(): full encode-decode pipeline
 *
 * Design Pattern (PyTorch-style):
 *   1. Subclass this base class
 *   2. Implement encode() and decode() 
 *   3. forward() calls encode() then decode() by default
 *   4. params() returns all trainable weights for optimizer
 *
 * @note This follows nn::Module<T> which provides:
 *         - state_dict() / load_state_dict() for serialization
 *         - params() for optimizer attachment
 */
#ifndef NN_MODELS_AUTOENCODER_BASE_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_BASE_AUTOENCODER_HPP

#include <span>
#include <vector>
#include <string>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @namespace nn::models::autoencoder
 * @brief Core autoencoder model implementations.
 *
 * Contains ANN and SNN autoencoders for various data modalities
 * (audio, EEG, fused). Designed to mirror PyTorch's nn.Module pattern.
 */
namespace nn::models::autoencoder
{

/**
 * @class BaseAutoencoder
 * @brief Abstract base class for all autoencoder architectures.
 *
 * Autoencoders learn to compress (encode) and reconstruct (decode) data.
 * The bottleneck (latent space) captures essential features.
 *
 * Architecture Overview:
 *   Input -> [Encoder] -> Latent -> [Decoder] -> Reconstruction
 *
 * @tparam Backend Tensor computation backend (e.g., nn::EigenTensorBackend)
 */
template <typename Backend>
class BaseAutoencoder : public nn::Module<Backend>
{
public:
    /**
     * @brief Virtual destructor for proper cleanup
     */
    ~BaseAutoencoder() override = default;

    /**
     * @brief Encode input to latent representation.
     *
     * The encoder transforms high-dimensional input into a lower-dimensional
     * latent representation. This is where compression happens.
     *
     * @param input Input tensor (batch_size x input_features)
     * @param requires_grad Whether to track gradients for this operation
     * @return Latent representation tensor (batch_size x latent_size)
     *
     * Example:
     *   @code
     *   nn::Tensor input = nn::Tensor::randn({32, 128});  // batch=32, features=128
     *   nn::Tensor latent = autoencoder.encode(input);
     *   // latent.shape() -> {32, 32}
     *   @endcode
     */
    virtual auto encode(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor = 0;

    /**
     * @brief Decode latent representation to reconstruction.
     *
     * The decoder transforms the latent representation back to the original
     * input dimension. Ideally, output ~= input (reconstruction).
     *
     * @param latent Latent tensor (batch_size x latent_size)
     * @param requires_grad Whether to track gradients for this operation
     * @return Reconstruction tensor (batch_size x input_features)
     */
    virtual auto decode(const nn::Tensor& latent, bool requires_grad = true) -> nn::Tensor = 0;

    /**
     * @brief Full encode-decode pipeline (forward pass).
     *
     * Default implementation: encode() -> decode()
     * Override this if you need custom forward logic.
     *
     * @param input Input tensor
     * @param requires_grad Whether to track gradients
     * @return Reconstruction tensor
     */
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Encode-compress the input to latent space
        nn::Tensor latent = encode(input, requires_grad);
        // Decode-expand back to original dimension
        return decode(latent, requires_grad);
    }

    /**
     * @brief Get architecture info for debugging/logging.
     * @return String describing the architecture type
     */
    virtual auto architecture_name() const -> std::string = 0;

protected:
    /**
     * @brief Constructor - initializes base class
     * @param name Module name for logging/serialization
     */
    explicit BaseAutoencoder(const std::string& name) : nn::Module<Backend>(name) {}
};

/**
 * @typedef AutoencoderTypes
 * @brief Convenience typedef for common autoencoder configurations
 *
 * - ANN: Standard artificial neural network (ReLU activations)
 * - SNN: Spiking neural network (LeakyIntegrator activations)
 * - Eigen: Eigen backend for CPU computation
 */
using EigenAutoencoder = BaseAutoencoder<nn::EigenTensorBackend>;
using EigenSpikingAutoencoder = BaseAutoencoder<nn::EigenTensorBackend>;

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_BASE_AUTOENCODER_HPP
