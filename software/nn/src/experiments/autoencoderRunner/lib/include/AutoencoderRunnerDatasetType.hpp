/**
 * @file src/experiments/autoencoderRunner/lib/include/AutoencoderRunnerDatasetType.hpp
 * @brief Dataset type enum for AutoencoderRunner.
 */

#pragma once

enum class AutoencoderRunnerDatasetType
{
    Protocol,
    EegWindow,
    AudioWindow,
    FusedWindow
};

inline auto dataset_type_to_string(AutoencoderRunnerDatasetType dataset_type) -> const char*
{
    switch (dataset_type)
    {
        case AutoencoderRunnerDatasetType::Protocol:
            return "protocol";
        case AutoencoderRunnerDatasetType::EegWindow:
            return "eeg-window";
        case AutoencoderRunnerDatasetType::AudioWindow:
            return "audio-window";
        case AutoencoderRunnerDatasetType::FusedWindow:
            return "fused-window";
    }

    return "unknown";
}
