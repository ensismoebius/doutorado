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

#include "nn/tensor/Tensor.hpp"

namespace nn::dataLoaders
{
class AudioMatSession
{
   public:
    explicit AudioMatSession(const std::string& filePath);
    ~AudioMatSession();

    AudioMatSession(const AudioMatSession&) = delete;
    AudioMatSession& operator=(const AudioMatSession&) = delete;
    AudioMatSession(AudioMatSession&&) noexcept;
    AudioMatSession& operator=(AudioMatSession&&) noexcept;

    auto readRow(size_t rowIndex) const -> std::tuple<nn::Tensor, int, int>;
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
