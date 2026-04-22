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
#include <stdexcept>
#include <string>

#include "Config.hpp"
#include "BaseAutoencoder.hpp"

// Forward declarations - implementations stay in experiments for now
// Core headers provide interface, experiments provide implementations
namespace nn::models::autoencoder
{
    class AudioWindowAutoencoder;
    class EegWindowAutoencoder;
    class FusedWindowAutoencoder;
}

/**
 * @namespace nn::models::autoencoder::builders
 * @brief Factory functions for autoencoder construction
 */
namespace nn::models::autoencoder::builders
{

/**
 * @enum AutoencoderType
 * @brief Supported autoencoder variants.
 *
 * Maps to experiment-specific implementations:
 *   - AudioWindow: Audio-only ANN (from Experiment03)
 *   - EegWindow: EEG-only ANN (from Experiment03)
 *   - FusedWindow: Multimodal ANN (from Experiment03)
 *   - AudioWindowSpiking: Audio-only SNN
 *   - EegWindowSpiking: EEG-only SNN
 *   - FusedWindowSpiking: Multimodal SNN
 *   - Protocol: Adaptive single/dual branch based on config
 */
enum class AutoencoderType
{
    AudioWindow,
    EegWindow,
    FusedWindow,
    AudioWindowSpiking,
    EegWindowSpiking,
    FusedWindowSpiking,
    Protocol,
    ProtocolSpiking
};

/**
 * @brief Convert string to AutoencoderType
 * @param type_str String like "audio", "eeg", "fused", "spiking", etc.
 * @return Corresponding AutoencoderType
 * @throws std::invalid_argument if unknown type
 */
inline auto from_string(const std::string& type_str) -> AutoencoderType
{
    if (type_str == "audio" || type_str == "AudioWindow")
        return AutoencoderType::AudioWindow;
    if (type_str == "eeg" || type_str == "EegWindow")
        return AutoencoderType::EegWindow;
    if (type_str == "fused" || type_str == "FusedWindow")
        return AutoencoderType::FusedWindow;
    if (type_str == "audio_snn" || type_str == "AudioWindowSpiking")
        return AutoencoderType::AudioWindowSpiking;
    if (type_str == "eeg_snn" || type_str == "EegWindowSpiking")
        return AutoencoderType::EegWindowSpiking;
    if (type_str == "fused_snn" || type_str == "FusedWindowSpiking")
        return AutoencoderType::FusedWindowSpiking;
    if (type_str == "protocol" || type_str == "Protocol")
        return AutoencoderType::Protocol;
    if (type_str == "protocol_snn" || type_str == "ProtocolSpiking")
        return AutoencoderType::ProtocolSpiking;

    throw std::invalid_argument("Unknown autoencoder type: " + type_str);
}

/**
 * @brief Create autoencoder from type enum
 *
 * This is the main factory function. Given a type and config,
 * it constructs the appropriate autoencoder.
 *
 * @param type Which autoencoder variant to create
 * @param cfg Configuration with hyperparameters
 * @return Unique pointer to created autoencoder
 *
 * @note Currently forwards to experiment implementations.
 *       Future: Move implementations to core/models/autoencoder/
 */
std::unique_ptr<BaseAutoencoder<nn::EigenTensorBackend>>
create(AutoencoderType type, const AutoencoderConfig& cfg);

/**
 * @brief Create autoencoder from string (convenience wrapper)
 * @param type_str Type as string (e.g., "audio", "fused")
 * @param cfg Configuration
 * @return Unique pointer to created autoencoder
 */
inline std::unique_ptr<BaseAutoencoder<nn::EigenTensorBackend>>
create(const std::string& type_str, const AutoencoderConfig& cfg)
{
    return create(from_string(type_str), cfg);
}

/**
 * @brief Get human-readable name for type
 * @param type Autoencoder type
 * @return String name (e.g., "AudioWindowAutoencoder")
 */
inline auto type_name(AutoencoderType type) -> std::string
{
    switch (type)
    {
        case AutoencoderType::AudioWindow:        return "AudioWindowAutoencoder";
        case AutoencoderType::EegWindow:          return "EegWindowAutoencoder";
        case AutoencoderType::FusedWindow:        return "FusedWindowAutoencoder";
        case AutoencoderType::AudioWindowSpiking: return "AudioWindowSpikingAutoencoder";
        case AutoencoderType::EegWindowSpiking:   return "EegWindowSpikingAutoencoder";
        case AutoencoderType::FusedWindowSpiking: return "FusedWindowSpikingAutoencoder";
        case AutoencoderType::Protocol:           return "ProtocolAutoencoder";
        case AutoencoderType::ProtocolSpiking:    return "ProtocolSpikingAutoencoder";
    }
    return "Unknown";
}

/**
 * @brief Check if type is an SNN (spiking) variant
 * @param type Autoencoder type
 * @return true if SNN variant
 */
inline auto is_snn(AutoencoderType type) -> bool
{
    return type == AutoencoderType::AudioWindowSpiking ||
           type == AutoencoderType::EegWindowSpiking ||
           type == AutoencoderType::FusedWindowSpiking ||
           type == AutoencoderType::ProtocolSpiking;
}

/**
 * @brief Check if type is multimodal (supports multiple inputs)
 * @param type Autoencoder type
 * @return true if multimodal
 */
inline auto is_multimodal(AutoencoderType type) -> bool
{
    return type == AutoencoderType::FusedWindow ||
           type == AutoencoderType::FusedWindowSpiking ||
           type == AutoencoderType::Protocol ||
           type == AutoencoderType::ProtocolSpiking;
}

} // namespace nn::models::autoencoder::builders

#endif // NN_MODELS_AUTOENCODER_BUILDERS_HPP
