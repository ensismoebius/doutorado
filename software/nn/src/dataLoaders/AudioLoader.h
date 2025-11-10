#pragma once

#include <Eigen/Dense>
#include <string>
#include <tuple>

namespace nn::dataLoaders
{

/**
 * @brief Loads audio data from a .mat file
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
