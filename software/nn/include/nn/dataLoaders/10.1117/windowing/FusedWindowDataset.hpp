/**
 * @file FusedWindowDataset.hpp
 * @brief Dataset that fuses synchronised EEG + audio windows into one input vector.
 *
 * Strategy — "Option 3" (Fused pipeline):
 *   For each (audio_row → EEG_row) matched recording pair in each subject,
 *   enumerate windows using *independent* `WindowSpec`s for each modality.
 *   The number of usable windows per pair is:
 *       N = min( n_windows(eeg_spec, eeg_samples),
 *                n_windows(audio_spec, audio_samples) )
 *   Window k from EEG and window k from audio are considered time-aligned
 *   (both start at position k * hop, expressed in their own sample coordinates).
 *
 * Fusion:
 *   z = [ x_EEG_flat ; x_audio_flat ]   (row-wise concatenation)
 *   where:
 *     x_EEG_flat  has shape (1, eeg_channels * eeg_window_size)   — channel-major.
 *     x_audio_flat has shape (1, audio_window_size).
 *   Combined input shape: (1, eeg_channels * eeg_window_size + audio_window_size).
 *
 * Output per sample:
 *   inputs : (1, eeg_channels * eeg_window_size + audio_window_size)
 *   targets: (1, 5) — [subject_id, eeg_label_0, eeg_label_1, eeg_label_2, eeg_index].
 *
 * Cross-modal synchronisation:
 *   The audio MAT row carries an `eeg_index_label` that identifies the simultaneously
 *   recorded EEG row.  This is read on each `get_item` call and resolved via the
 *   existing `resolveEegRowIndex` helper (SchemaIndexing.hpp).
 *
 * Thread safety: NOT thread-safe.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/loaders/AudioLoader.h"
#include "nn/dataLoaders/10.1117/loaders/EEGLoader.h"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/Dataset.hpp"
#include "nn/windowing/WindowSpec.hpp"

class FusedWindowDataset : public Dataset
{
   public:
    /**
     * @param subjects    Subjects to include.
     * @param eeg_spec    Windowing spec for EEG (window_size in EEG samples).
     * @param audio_spec  Windowing spec for audio (window_size in audio samples).
     *
     * The two specs are independent but must produce a non-zero number of windows
     * from their respective signals (EEG: 4 096 samples; audio: 176 400 samples).
     */
    explicit FusedWindowDataset(std::vector<SubjectFiles> subjects,
        nn::windowing::WindowSpec eeg_spec,
        nn::windowing::WindowSpec audio_spec);

    /// Total number of fused windows across all subjects and rows.
    [[nodiscard]] auto size() const -> std::size_t override;

    /**
     * @brief Retrieve a single fused (EEG + audio) windowed sample.
     * @param idx  Global window index in [0, size()).
     * @return Batch with:
     *   .inputs  (1, eeg_channels * eeg_window_size + audio_window_size)
     *   .targets (1, 5) — [subject_id, eeg_lbl[0], eeg_lbl[1], eeg_lbl[2], eeg_index].
     */
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;

    /// Efficient batch assembly.
    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override;

    [[nodiscard]] auto eeg_spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return eeg_spec_;
    }
    [[nodiscard]] auto audio_spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return audio_spec_;
    }

    /// Number of usable fused windows per matched recording pair.
    [[nodiscard]] auto windows_per_pair() const noexcept -> int
    {
        return windows_per_pair_;
    }

    /// Flat size of one fused input vector (eeg_channels * eeg_ws + audio_ws).
    [[nodiscard]] auto input_features() const noexcept -> int
    {
        return input_features_;
    }

   private:
    /// One entry in the flat index table.
    struct WindowIndex
    {
        std::size_t subject_idx; ///< Into `subjects_`.
        std::size_t audio_row;   ///< Local audio row within subject.
        int window_k;            ///< 0-based window index within this recording.
    };

    void ensure_sessions(std::size_t subject_idx) const;

    std::vector<SubjectFiles> subjects_;
    nn::windowing::WindowSpec eeg_spec_;
    nn::windowing::WindowSpec audio_spec_;
    int windows_per_pair_{0};
    int input_features_{0};

    std::vector<WindowIndex> index_table_;

    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>> eeg_sessions_;
};
