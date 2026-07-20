/**
 * @file src/experiments/autoencoderRunner/lib/include/Experiment03DatasetType.hpp
 * @brief Dataset type enum for Experiment03.
 */

#pragma once

enum class Experiment03DatasetType
{
    Protocol,
    EegWindow,
    AudioWindow,
    FusedWindow
};

inline auto dataset_type_to_string(Experiment03DatasetType dataset_type) -> const char*
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return "protocol";
        case Experiment03DatasetType::EegWindow:
            return "eeg-window";
        case Experiment03DatasetType::AudioWindow:
            return "audio-window";
        case Experiment03DatasetType::FusedWindow:
            return "fused-window";
    }

    return "unknown";
}
