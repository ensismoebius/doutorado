/**
 * @file MatFileFlags.h
 * @brief Column/field indices for MAT dataset layouts used by the loaders.
 *
 * The codebase uses these enums as self-documenting alternatives to “magic indices”.
 */

#pragma once

#include <cstdint>
namespace MatFileFlags
{

enum struct EEGFlag : uint8_t
{
    Modality = 0, // 0-based index for the eegInfo array
    Stimulus = 1, // 0-based index for the eegInfo array
    Artifact = 2  // 0-based index for the eegInfo array
};

enum struct AudioFlag : uint32_t
{
    Stimulus = 176400, // 0-based index for Audio(row, 176401)
    EEG_Index = 176401 // 0-based index for Audio(row, 176402)
};

// Helper function to get string representation of EEGFlag
inline auto getEEGFlagName(EEGFlag flag) -> const char*
{
    switch (flag)
    {
        case EEGFlag::Modality:
            return "Modality";
        case EEGFlag::Stimulus:
            return "Stimulus";
        case EEGFlag::Artifact:
            return "Artifact";
        default:
            return "Unknown EEG Flag";
    }
}

// Helper function to get string representation of AudioFlag
inline auto getAudioFlagName(AudioFlag flag) -> const char*
{
    switch (flag)
    {
        case AudioFlag::Stimulus:
            return "Stimulus";
        case AudioFlag::EEG_Index:
            return "EEG_Index";
        default:
            return "Unknown Audio Flag";
    }
}

} // namespace MatFileFlags
