/**
 * @file AudioLoader.h
 * @brief Helpers for loading audio rows from the 10.1117-style MAT dataset into `nn::Tensor`.
 *
 * Contract (current dataset convention):
 * - Audio signal is stored as a single long row of samples followed by label columns.
 * - `loadAudioFromMat()` returns a column-vector tensor of raw samples plus two integer labels.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "nn/tensor/Tensor.hpp"

namespace nn::dataLoaders
{
struct AudioRowsFlat
{
    std::vector<float> samples; // row-major: [row0_samples..., row1_samples...]
    std::vector<int> stimuli;
    std::vector<int> eegIndices;
};

class AudioMatSession
{
   public:
    // `filePath` may be either a path to a .mat file or a .sqlite database.
    // When using a sqlite DB, callers should provide `subject_id` so the
    // session can scope queries to a single subject. The parameter has a
    // default of -1 to preserve backward compatibility with existing calls.
    explicit AudioMatSession(const std::string& filePath, int subject_id = -1);
    ~AudioMatSession();

    AudioMatSession(const AudioMatSession&) = delete;
    AudioMatSession& operator=(const AudioMatSession&) = delete;
    AudioMatSession(AudioMatSession&&) noexcept;
    AudioMatSession& operator=(AudioMatSession&&) noexcept;

    auto readRow(size_t rowIndex) const -> std::tuple<nn::Tensor, int, int>;
    auto readRows(size_t startRow, size_t rowCount) const
        -> std::vector<std::tuple<nn::Tensor, int, int>>;
    auto readRowsFlat(size_t startRow, size_t rowCount) const -> AudioRowsFlat;
    [[nodiscard]] auto rowCount() const -> size_t;
    [[nodiscard]] auto filePath() const -> const std::string&;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Load audio data from a MAT file. Searches for the variable named "Audio" and extracts the
 * audio samples (which starts at the column 0 and ends at column 176400) and EEG index (which is
 * located in the last column of the matrix).
 *
 * @param filePath Path to the .mat file
 * @param rowIndex The row index to load (defaults to 0)
 * @return std::tuple containing:
 *         - nn::Tensor: Audio samples (176400 samples @ 44100 Hz) as column vector (176400, 1)
 *         - int: Stimulus (e.g., word spoken)
 *         - int: EEG index (corresponding EEG row where this audio was recorded)
 * @throws std::runtime_error if file cannot be opened or has invalid format
 */
auto loadAudioFromMat(const std::string& filePath, size_t rowIndex = 0)
    -> std::tuple<nn::Tensor, int, int>;

} // namespace nn::dataLoaders
