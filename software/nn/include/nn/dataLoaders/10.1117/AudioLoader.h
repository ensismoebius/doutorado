#pragma once

#include <map>
#include <string>
#include <tuple>

#include "nn/tensor/Tensor.hpp"

namespace nn::dataLoaders
{

/**
 * @brief Map stimulus IDs to their string representations.
 * For this dataset, the stimulus are A, E, I, O, U vowels
 * and directions in Spanish and the words "Up", "Down", "Forward",
 * "Backward", "Right", "Left" in Spanish.
 */
const std::map<int, std::string> ESTIMULUS_NAMES = {
    {1, "A"},         // Vowel "A"
    {2, "E"},         // Vowel "E"
    {3, "I"},         // Vowel "I"
    {4, "O"},         // Vowel "O"
    {5, "U"},         // Vowel "U"
    {6, "Arriba"},    // "Up" in Spanish
    {7, "Abajo"},     //  "Down" in Spanish
    {8, "Adelante"},  // "Forward" in Spanish
    {9, "Atras"},     // "Backward" in Spanish
    {10, "Derecha"},  // "Right" in Spanish
    {11, "Izquierda"} // "Left" in Spanish
};

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
 *         - nn::Tensor: Audio samples (176400 samples @ 44100 Hz) as column vector (176400, 1)
 *         - int: Stimulus (e.g., word spoken)
 *         - int: EEG index (corresponding EEG row where this audio was recorded)
 * @throws std::runtime_error if file cannot be opened or has invalid format
 */
auto loadAudioFromMat(const std::string& filePath, size_t rowIndex = 0)
    -> std::tuple<nn::Tensor, int, int>;

} // namespace nn::dataLoaders
