/**
 * @file src/experiments/autoencoderRunner/lib/include/AutoencoderRunnerAutoencoderType.hpp
 * @brief Autoencoder type enum and helpers for AutoencoderRunner.
 */

#pragma once

#include "AutoencoderRunnerDatasetType.hpp"

enum class AutoencoderRunnerAutoencoderType
{
    ProtocolAnn,
    EegWindowAnn,
    AudioWindowAnn,
    FusedWindowAnn,
    ProtocolSnn,
    EegWindowSnn,
    AudioWindowSnn,
    FusedWindowSnn
};

inline auto autoencoder_type_to_string(AutoencoderRunnerAutoencoderType autoencoder_type) -> const
    char*
{
    switch (autoencoder_type)
    {
        case AutoencoderRunnerAutoencoderType::ProtocolAnn:
            return "protocol-ann";
        case AutoencoderRunnerAutoencoderType::EegWindowAnn:
            return "eeg-window-ann";
        case AutoencoderRunnerAutoencoderType::AudioWindowAnn:
            return "audio-window-ann";
        case AutoencoderRunnerAutoencoderType::FusedWindowAnn:
            return "fused-window-ann";
        case AutoencoderRunnerAutoencoderType::ProtocolSnn:
            return "protocol-snn";
        case AutoencoderRunnerAutoencoderType::EegWindowSnn:
            return "eeg-window-snn";
        case AutoencoderRunnerAutoencoderType::AudioWindowSnn:
            return "audio-window-snn";
        case AutoencoderRunnerAutoencoderType::FusedWindowSnn:
            return "fused-window-snn";
    }

    return "unknown";
}

inline auto is_autoencoder_compatible(AutoencoderRunnerDatasetType dataset_type,
    AutoencoderRunnerAutoencoderType autoencoder_type) -> bool
{
    switch (dataset_type)
    {
        case AutoencoderRunnerDatasetType::Protocol:
            return autoencoder_type == AutoencoderRunnerAutoencoderType::ProtocolAnn ||
                   autoencoder_type == AutoencoderRunnerAutoencoderType::ProtocolSnn;
        case AutoencoderRunnerDatasetType::EegWindow:
            return autoencoder_type == AutoencoderRunnerAutoencoderType::EegWindowAnn ||
                   autoencoder_type == AutoencoderRunnerAutoencoderType::EegWindowSnn;
        case AutoencoderRunnerDatasetType::AudioWindow:
            return autoencoder_type == AutoencoderRunnerAutoencoderType::AudioWindowAnn ||
                   autoencoder_type == AutoencoderRunnerAutoencoderType::AudioWindowSnn;
        case AutoencoderRunnerDatasetType::FusedWindow:
            return autoencoder_type == AutoencoderRunnerAutoencoderType::FusedWindowAnn ||
                   autoencoder_type == AutoencoderRunnerAutoencoderType::FusedWindowSnn;
    }

    return false;
}

inline auto is_snn_type(AutoencoderRunnerAutoencoderType t) -> bool
{
    return t == AutoencoderRunnerAutoencoderType::ProtocolSnn ||
           t == AutoencoderRunnerAutoencoderType::EegWindowSnn ||
           t == AutoencoderRunnerAutoencoderType::AudioWindowSnn ||
           t == AutoencoderRunnerAutoencoderType::FusedWindowSnn;
}
