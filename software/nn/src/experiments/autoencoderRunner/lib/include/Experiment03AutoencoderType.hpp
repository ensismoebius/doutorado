/**
 * @file src/experiments/autoencoderRunner/lib/include/Experiment03AutoencoderType.hpp
 * @brief Autoencoder type enum and helpers for Experiment03.
 */

#pragma once

#include "Experiment03DatasetType.hpp"

enum class Experiment03AutoencoderType
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

inline auto autoencoder_type_to_string(Experiment03AutoencoderType autoencoder_type) -> const char*
{
    switch (autoencoder_type)
    {
        case Experiment03AutoencoderType::ProtocolAnn:
            return "protocol-ann";
        case Experiment03AutoencoderType::EegWindowAnn:
            return "eeg-window-ann";
        case Experiment03AutoencoderType::AudioWindowAnn:
            return "audio-window-ann";
        case Experiment03AutoencoderType::FusedWindowAnn:
            return "fused-window-ann";
        case Experiment03AutoencoderType::ProtocolSnn:
            return "protocol-snn";
        case Experiment03AutoencoderType::EegWindowSnn:
            return "eeg-window-snn";
        case Experiment03AutoencoderType::AudioWindowSnn:
            return "audio-window-snn";
        case Experiment03AutoencoderType::FusedWindowSnn:
            return "fused-window-snn";
    }

    return "unknown";
}

inline auto is_autoencoder_compatible(
    Experiment03DatasetType dataset_type, Experiment03AutoencoderType autoencoder_type) -> bool
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return autoencoder_type == Experiment03AutoencoderType::ProtocolAnn ||
                   autoencoder_type == Experiment03AutoencoderType::ProtocolSnn;
        case Experiment03DatasetType::EegWindow:
            return autoencoder_type == Experiment03AutoencoderType::EegWindowAnn ||
                   autoencoder_type == Experiment03AutoencoderType::EegWindowSnn;
        case Experiment03DatasetType::AudioWindow:
            return autoencoder_type == Experiment03AutoencoderType::AudioWindowAnn ||
                   autoencoder_type == Experiment03AutoencoderType::AudioWindowSnn;
        case Experiment03DatasetType::FusedWindow:
            return autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
                   autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn;
    }

    return false;
}

inline auto is_snn_type(Experiment03AutoencoderType t) -> bool
{
    return t == Experiment03AutoencoderType::ProtocolSnn ||
           t == Experiment03AutoencoderType::EegWindowSnn ||
           t == Experiment03AutoencoderType::AudioWindowSnn ||
           t == Experiment03AutoencoderType::FusedWindowSnn;
}
