/**
 * @file EEGWindowDataset.hpp
 * @brief Dataset that slices EEG recordings into overlapping windows.
 *
 * Strategy — "Option 2" (EEG pipeline):
 *   For each subject, for each recording row, enumerate all complete sliding
 *   windows defined by `WindowSpec`.  The dataset index space is the flat
 *   concatenation of all (subject × row × window) triples.
 *
 * Output shape per sample:
 *   inputs : (1, eeg_channels × eeg_window_size)  — channel-major, flattened.
 *   targets: (1, 3)                                — [subject_id, stimulus, blink].
 *
 * EEG tensor from `EEGMatSession::readRow` has shape (eeg_channels, eeg_samples).
 * A window of `window_size` time samples is extracted as:
 *   out[0, c * window_size + t] = eeg[c, window_start + t]  for c in channels, t in window.
 *
 * MAT sessions are opened lazily (first access per subject), matching the
 * behaviour of `Protocol101117Dataset`.
 *
 * Thread safety: NOT thread-safe.  Use one dataset instance per thread or
 * add external synchronization (mirrors matio constraints in the project).
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/loaders/EEGLoader.h"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/Dataset.hpp"
#include "nn/windowing/WindowSpec.hpp"

class EEGWindowDataset : public Dataset
{
   public:
    /**
     * @param subjects   Subjects to include.  Each provides a path to the EEG
     *                   MAT file and the total number of EEG rows in that file.
     * @param spec       Windowing spec:  window_size in EEG samples (e.g. 256
     *                   for ~250 ms at 1 024 Hz), overlap in [0, 1).
     */
    explicit EEGWindowDataset(std::vector<SubjectFiles> subjects, nn::windowing::WindowSpec spec);

    /// Total number of windows across all subjects and rows.
    [[nodiscard]] auto size() const -> std::size_t override;

    /**
     * @brief Retrieve a single windowed EEG sample.
     * @param idx  Global window index in [0, size()).
     * @return Batch with:
     *   .inputs  (1, eeg_channels * window_size) — windowed EEG, channel-major.
     *   .targets (1, 3)                          — [subject_id, stimulus, blink].
     */
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;

    /// Efficient batch assembly; pre-allocates batch tensors on first call.
    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override;

    /// Read-only access to the windowing spec.
    [[nodiscard]] auto spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return spec_;
    }

    /// Number of windows per EEG row (constant for all rows given a fixed schema).
    [[nodiscard]] auto windows_per_row() const noexcept -> int
    {
        return windows_per_row_;
    }

   private:
    /// Index entry describing one window's position in the dataset.
    struct WindowIndex
    {
        std::size_t subject_idx; ///< Into `subjects_`.
        std::size_t row_idx;     ///< Local EEG row within the subject.
        int window_start;        ///< Start sample index within the row.
    };

    /// Lazy‐init the EEG MAT session for the given subject.
    void ensure_session(std::size_t subject_idx) const;

    std::vector<SubjectFiles> subjects_;
    nn::windowing::WindowSpec spec_;
    int windows_per_row_{0};

    /// Flat window index table, built in constructor.
    std::vector<WindowIndex> index_table_;

    /// Lazily-opened EEG sessions, one per subject.
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>> eeg_sessions_;
};
