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
inline const std::map<int, std::string> kStimulusNames = {
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

inline const std::map<int, std::string> kArtifactNames = {
    {1, "No blink"},
    {2, "Blink"},
};

inline const std::map<int, std::string> kModalityNames = {
    {1, "Imagined"},
    {2, "Pronounced"},
};

// EEG channel names corresponding to the 6 channels in the dataset
inline const std::array<std::string, 6> kEegChannelNames = {"F3", "F4", "C3", "C4", "P3", "P4"};

inline const std::string kEegMatFileSuffix = "_EEG.mat";
inline const std::string kAudioMatFileSuffix = "_Audio.mat";

inline const std::string kEegMatVariableName = "EEG";
inline const std::string kAudioMatVariableName = "Audio";

} // namespace nn::dataLoaders