/**
 * @file include/nn/dataLoaders/10.1117/schema/METADATA.hpp
 * @brief Metadata.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#pragma once

#include <cstddef>

namespace nn::dataLoaders
{

/*
Dataset schema for the "Imagined Speech EEG Database".

Source:
"Open access database of EEG signals recorded during imagined speech"
Pressel-Coretto, Gareis, Rufiner (SPIE 2017)

Key protocol details from the article:

• Each stimulus trial contains a 4-second Imagine/Pronounce interval.
• EEG was sampled at 1024 Hz using six channels:
  F3, F4, C3, C4, P3, P4.
• The six EEG channels are concatenated into a single vector.
• Three labels are appended to each EEG vector:
      1) mode (imagined or pronounced speech)
      2) stimulus code (word or vowel)
      3) blink artifact flag

• Audio was recorded only in the pronounced speech condition.
• Audio was sampled at 44.1 kHz with a single channel.
• Two labels are appended to the audio vector:
      1) stimulus code
      2) index of the corresponding EEG row
*/

struct DatasetSchema
{
    // ============================================================
    // Acquisition parameters defined in the experimental protocol
    // ============================================================

    // Duration of the Imagine/Pronounce interval (seconds)
    // Each recorded trial corresponds to this 4-second window.
    size_t duration_seconds;

    // EEG acquisition parameters
    size_t eeg_sampling_rate; // 1024 Hz
    size_t eeg_channels;      // 6 electrodes: F3 F4 C3 C4 P3 P4
    size_t eeg_label_columns; // mode, stimulus code, blink artifact

    // Audio acquisition parameters
    size_t audio_sampling_rate; // 44.1 kHz
    size_t audio_label_columns; // stimulus code, EEG row index

    // ============================================================
    // Derived EEG dimensions
    // ============================================================

    // Number of samples per channel for one trial
    // samples = sampling_rate × duration
    constexpr size_t eegSamplesPerChannel() const
    {
        return eeg_sampling_rate * duration_seconds; // 1024 × 4 = 4096
    }

    // Number of columns containing EEG signal values
    // channels concatenated sequentially
    constexpr size_t eegSignalColumns() const
    {
        return eegSamplesPerChannel() * eeg_channels; // 4096 × 6 = 24576
    }

    // Total columns in EEG matrix row
    // signal columns + metadata labels
    constexpr size_t eegTotalColumns() const
    {
        return eegSignalColumns() + eeg_label_columns; // 24579
    }

    // Column index of the speech mode label
    // (imagined or pronounced)
    constexpr size_t eegModeColumn() const
    {
        return eegSignalColumns();
    }

    // Column index of the stimulus identifier
    // (vowel or command word)
    constexpr size_t eegStimulusColumn() const
    {
        return eegSignalColumns() + 1;
    }

    // Column index of the blink artifact flag
    constexpr size_t eegBlinkColumn() const
    {
        return eegSignalColumns() + 2;
    }

    // ============================================================
    // Derived Audio dimensions
    // ============================================================

    // Number of audio samples for one trial
    // samples = sampling_rate × duration
    constexpr size_t audioSamples() const
    {
        return audio_sampling_rate * duration_seconds; // 44100 × 4 = 176400
    }

    // Total columns in audio matrix row
    constexpr size_t audioTotalColumns() const
    {
        return audioSamples() + audio_label_columns; // 176402
    }

    // Column index containing the stimulus label
    constexpr size_t audioStimulusColumn() const
    {
        return audioSamples();
    }

    // Column index referencing the EEG row recorded simultaneously
    constexpr size_t audioEEGIndexColumn() const
    {
        return audioSamples() + 1;
    }
};

// ============================================================
// Concrete schema instance for the imagined speech database
// ============================================================

constexpr DatasetSchema ImaginedSpeechSchema_10_1117{.duration_seconds = 4,

    // EEG parameters described in Section 2.3 (Data Collection)
    .eeg_sampling_rate = 1024,
    .eeg_channels = 6,
    .eeg_label_columns = 3,

    // Audio acquisition parameters
    .audio_sampling_rate = 44100,
    .audio_label_columns = 2};

} // namespace nn::dataLoaders