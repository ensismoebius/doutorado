#pragma once

#include <Eigen/Dense>
#include <string>
#include <tuple>

namespace nn::dataLoaders
{

// Audio data format constants
constexpr size_t AUDIO_SAMPLES_COUNT = 176400; // Number of audio samples per row
constexpr size_t MATRIX_COLUMNS = 176402;      // Total columns (samples + stimulus + EEG index)
constexpr size_t STIMULUS_COLUMN = 176400;     // Column index for stimulus
constexpr size_t EEG_INDEX_COLUMN = 176401;    // Column index for EEG index
constexpr const char* AUDIO_VARIABLE_NAME = "Audio"; // Name of the variable in MAT file

/**
 * @brief Load audio data from a MAT file. Searches for the variable named "Audio" and extracts the
 * audio samples (which starts at the column 0 and ends at column 176400) and EEG index (which is
 * located in the last column of the matrix).
 *
 * @param filePath Path to the .mat file
 * @param rowIndex The row index to load (defaults to 0)
 * @return std::tuple containing:
 *         - Eigen::VectorXf: Audio samples (176400 samples @ 44100 Hz)
 *         - int: EEG index (corresponding EEG row where this audio was recorded)
 * @throws std::runtime_error if file cannot be opened or has invalid format
 */
auto loadAudioFromMat(const std::string& filePath, size_t rowIndex = 0)
    -> std::tuple<Eigen::VectorXf, int>;

} // namespace nn::dataLoaders
