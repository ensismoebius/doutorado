#include <array>
#include <map>
#include <string>

namespace nn::dataLoaders
{
// Each table is a function returning a function-local static: a namespace-scope std::map or
// std::string may throw while statics are initialised, where nothing can catch it.
/**
 * @brief Map stimulus IDs to their string representations.
 * For this dataset, the stimulus are A, E, I, O, U vowels
 * and directions in Spanish and the words "Up", "Down", "Forward",
 * "Backward", "Right", "Left" in Spanish.
 */
inline auto kStimulusNames() -> const std::map<int, std::string>&
{
    static const std::map<int, std::string> value = {
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
    return value;
}

inline auto kArtifactNames() -> const std::map<int, std::string>&
{
    static const std::map<int, std::string> value = {
        {1, "No blink"},
        {2, "Blink"},
    };
    return value;
}

inline auto kModalityNames() -> const std::map<int, std::string>&
{
    static const std::map<int, std::string> value = {
        {1, "Imagined"},
        {2, "Pronounced"},
    };
    return value;
}

// EEG channel names corresponding to the 6 channels in the dataset
inline auto kEegChannelNames() -> const std::array<std::string, 6>&
{
    static const std::array<std::string, 6> value = {"F3", "F4", "C3", "C4", "P3", "P4"};
    return value;
}

inline auto kEegMatFileSuffix() -> const std::string&
{
    static const std::string value = "_EEG.mat";
    return value;
}
inline auto kAudioMatFileSuffix() -> const std::string&
{
    static const std::string value = "_Audio.mat";
    return value;
}

inline auto kEegMatVariableName() -> const std::string&
{
    static const std::string value = "EEG";
    return value;
}
inline auto kAudioMatVariableName() -> const std::string&
{
    static const std::string value = "Audio";
    return value;
}

} // namespace nn::dataLoaders