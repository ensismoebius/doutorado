/**
 * @file AudioWindowDataset.hpp
 * @brief Dataset that slices audio recordings into overlapping windows.
 *
 * Strategy — "Option 2" (Audio pipeline):
 *   For each subject, for each audio recording row, enumerate all complete
 *   sliding windows defined by `WindowSpec`.  The index space is the flat
 *   concatenation of all (subject × row × window) triples.
 *
 * Output shape per sample:
 *   inputs : (1, audio_window_size) — raw audio samples.
 *   targets: (1, 2)                 — [stimulus, eeg_index_label].
 *
 * Audio tensor from `AudioMatSession::readRow` has shape (audio_samples, 1)
 * (a column vector).  The windowed output extracts rows [start, start+W) and
 * reshapes them into a flat row vector.
 *
 * MAT sessions are opened lazily (first access per subject), matching the
 * behaviour of `Protocol101117Dataset`.
 *
 * Thread safety: NOT thread-safe.  (See EEGWindowDataset.hpp note.)
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/loaders/AudioLoader.h"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/Dataset.hpp"
#include "nn/windowing/WindowSpec.hpp"

class AudioWindowDataset : public Dataset
{
   public:
    /**
     * @param subjects   Subjects to include.  Each provides a path to the audio
     *                   MAT file and the total number of audio rows.
     * @param spec       Windowing spec: window_size in audio samples (e.g. 11 025
     *                   for ~250 ms at 44 100 Hz), overlap in [0, 1).
     */
    explicit AudioWindowDataset(std::vector<SubjectFiles> subjects, nn::windowing::WindowSpec spec);

    /// Total number of windows across all subjects and rows.
    [[nodiscard]] auto size() const -> std::size_t override;

    /**
     * @brief Retrieve a single windowed audio sample.
     * @param idx  Global window index in [0, size()).
     * @return Batch with:
     *   .inputs  (1, window_size) — raw audio window.
     *   .targets (1, 2)           — [stimulus, eeg_index_label].
     */
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;

    /// Efficient batch assembly.
    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override;

    /// Read-only access to the windowing spec.
    [[nodiscard]] auto spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return spec_;
    }

    /// Number of windows per audio row (constant for fixed schema).
    [[nodiscard]] auto windows_per_row() const noexcept -> int
    {
        return windows_per_row_;
    }

   private:
    struct WindowIndex
    {
        std::size_t subject_idx;
        std::size_t row_idx;
        int window_start;
    };

    void ensure_session(std::size_t subject_idx) const;

    std::vector<SubjectFiles> subjects_;
    nn::windowing::WindowSpec spec_;
    int windows_per_row_{0};

    std::vector<WindowIndex> index_table_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
};
