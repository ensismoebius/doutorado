#include <array>
#include <map>
#include <string>

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

const std::map<int, std::string> ARTIFACT_NAMES = {
    {1, "No blink"},
    {2, "Blink"},
};

const std::map<int, std::string> MODALITY_NAMES = {
    {1, "Imagined"},
    {2, "Pronounced"},
};

// EEG channel names corresponding to the 6 channels in the dataset
constexpr std::array<std::string, 6> EEG_CHANNELS_NAMES = {"F3", "F4", "C3", "C4", "P3", "P4"};

constexpr std::string EEG_MAT_FILE_SUFFIX = "_EEG.mat";
constexpr std::string AUDIO_MAT_FILE_SUFFIX = "_Audio.mat";

constexpr std::string EEG_MAT_VARIABLE_NAME = "EEG";
constexpr std::string AUDIO_MAT_VARIABLE_NAME = "Audio";

} // namespace nn::dataLoaders