/**
 * @file AutoencoderBuilders.hpp
 * @brief Factory for creating autoencoder instances.
 *
 * This builder follows the Factory Pattern (GoF) and PyTorch's model
 * construction style. Instead of directly instantiating autoencoders,
 * use these builders for cleaner, configurable construction.
 *
 * Design Pattern:
 *   Instead of:
 *     auto model = AudioWindowAutoencoder(cfg);
 *   Use:
 *     auto model = AutoencoderBuilders::create("audio", cfg);
 *
 * Benefits:
 *   - Centralized creation logic
 *   - Easy to add new types
 *   - Runtime type selection from config strings
 *   - Consistent error handling
 *
 * Example Usage:
 *   @code
 *   // Create by type string
 *   auto audio_ae = AutoencoderBuilders::create(AutoencoderType::AudioWindow, config);
 *   auto eeg_ae = AutoencoderBuilders::create(AutoencoderType::EegWindow, config);
 *   auto fused_ae = AutoencoderBuilders::create(AutoencoderType::FusedWindow, config);
 *
 *   // Create SNN variants
 *   auto audio_snn = AutoencoderBuilders::create(AudioWindowSpiking, config);
 *   @endcode
 */
#ifndef NN_MODELS_AUTOENCODER_BUILDERS_HPP
#define NN_MODELS_AUTOENCODER_BUILDERS_HPP

#include <memory>

#include "AutoencoderType.hpp"
#include "BaseAutoencoder.hpp"
#include "Config.hpp"

// Forward declarations - implementations stay in experiments for now
// Core headers provide interface, experiments provide implementations
namespace nn::models::autoencoder
{
class AudioWindowAutoencoder;
class EegWindowAutoencoder;
class FusedWindowAutoencoder;
} // namespace nn::models::autoencoder

namespace nn::models::autoencoder::builders
{

std::unique_ptr<BaseAutoencoder<nn::EigenTensorBackend>> create(
    AutoencoderType type, const AutoencoderConfig& cfg);

inline std::unique_ptr<BaseAutoencoder<nn::EigenTensorBackend>> create(
    const std::string& type_str, const AutoencoderConfig& cfg)
{
    return create(from_string(type_str), cfg);
}

} // namespace nn::models::autoencoder::builders

#endif // NN_MODELS_AUTOENCODER_BUILDERS_HPP
