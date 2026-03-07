#include <cstddef>
namespace nn::dataLoaders
{

// Constants matching dataset schema for Audio
constexpr size_t AUDIO_SAMPLES_COUNT = 176400; // Number of audio samples per row
constexpr size_t MATRIX_COLUMNS = 176402;      // Total columns (samples + stimulus + EEG index)
constexpr size_t STIMULUS_COLUMN = 176400;     // Column index for stimulus
constexpr size_t EEG_INDEX_COLUMN = 176401;    // Column index for EEG index
constexpr const char* AUDIO_VARIABLE_NAME = "Audio"; // Name of the variable in MAT file

// Constants matching dataset schema for EEG
constexpr int EEG_TOTAL_COLUMNS = 24579; // M columns including labels
constexpr int EEG_SAMPLE_COUNT = 24576;  // samples portion
constexpr int EEG_CHANNELS = 6;          // number of EEG channels

} // namespace nn::dataLoaders
