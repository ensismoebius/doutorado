#pragma once

namespace MatFileFlags {

enum class EEGFlag {
    Modality = 0, // 0-based index for the eegInfo array
    Stimulus = 1, // 0-based index for the eegInfo array
    Artifact = 2  // 0-based index for the eegInfo array
};

enum class AudioFlag {
    Stimulus = 176400, // 0-based index for Audio(row, 176401)
    EEG_Index = 176401 // 0-based index for Audio(row, 176402)
};

// Helper function to get string representation of EEGFlag
inline const char* getEEGFlagName(EEGFlag flag) {
    switch (flag) {
        case EEGFlag::Modality: return "Modality";
        case EEGFlag::Stimulus: return "Stimulus";
        case EEGFlag::Artifact: return "Artifact";
        default: return "Unknown EEG Flag";
    }
}

// Helper function to get string representation of AudioFlag
inline const char* getAudioFlagName(AudioFlag flag) {
    switch (flag) {
        case AudioFlag::Stimulus: return "Stimulus";
        case AudioFlag::EEG_Index: return "EEG_Index";
        default: return "Unknown Audio Flag";
    }
}

} // namespace MatFileFlags
