#ifndef NN_MODELS_AUTOENCODER_TYPE_HPP
#define NN_MODELS_AUTOENCODER_TYPE_HPP

#include <stdexcept>
#include <string>

namespace nn::models::autoencoder::builders
{

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

inline auto from_string(const std::string& type_str) -> AutoencoderType
{
    if (type_str == "audio" || type_str == "AudioWindow") return AutoencoderType::AudioWindow;
    if (type_str == "eeg" || type_str == "EegWindow") return AutoencoderType::EegWindow;
    if (type_str == "fused" || type_str == "FusedWindow") return AutoencoderType::FusedWindow;
    if (type_str == "audio_snn" || type_str == "AudioWindowSpiking")
        return AutoencoderType::AudioWindowSpiking;
    if (type_str == "eeg_snn" || type_str == "EegWindowSpiking")
        return AutoencoderType::EegWindowSpiking;
    if (type_str == "fused_snn" || type_str == "FusedWindowSpiking")
        return AutoencoderType::FusedWindowSpiking;
    if (type_str == "protocol" || type_str == "Protocol") return AutoencoderType::Protocol;
    if (type_str == "protocol_snn" || type_str == "ProtocolSpiking")
        return AutoencoderType::ProtocolSpiking;

    throw std::invalid_argument("Unknown autoencoder type: " + type_str);
}

inline auto type_name(AutoencoderType type) -> std::string
{
    switch (type)
    {
        case AutoencoderType::AudioWindow:
            return "AudioWindowAutoencoder";
        case AutoencoderType::EegWindow:
            return "EegWindowAutoencoder";
        case AutoencoderType::FusedWindow:
            return "FusedWindowAutoencoder";
        case AutoencoderType::AudioWindowSpiking:
            return "AudioWindowSpikingAutoencoder";
        case AutoencoderType::EegWindowSpiking:
            return "EegWindowSpikingAutoencoder";
        case AutoencoderType::FusedWindowSpiking:
            return "FusedWindowSpikingAutoencoder";
        case AutoencoderType::Protocol:
            return "ProtocolAutoencoder";
        case AutoencoderType::ProtocolSpiking:
            return "ProtocolSpikingAutoencoder";
    }
    return "Unknown";
}

inline auto is_snn(AutoencoderType type) -> bool
{
    return type == AutoencoderType::AudioWindowSpiking ||
           type == AutoencoderType::EegWindowSpiking ||
           type == AutoencoderType::FusedWindowSpiking || type == AutoencoderType::ProtocolSpiking;
}

inline auto is_multimodal(AutoencoderType type) -> bool
{
    return type == AutoencoderType::FusedWindow || type == AutoencoderType::FusedWindowSpiking ||
           type == AutoencoderType::Protocol || type == AutoencoderType::ProtocolSpiking;
}

} // namespace nn::models::autoencoder::builders

#endif // NN_MODELS_AUTOENCODER_TYPE_HPP