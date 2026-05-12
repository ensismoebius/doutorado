#pragma once
// Names.hpp — FSDD speaker list and digit labels (DOI: 10.5281/zenodo.1342401).

#include <array>
#include <string_view>

namespace nn::dataLoaders::fsdd
{

// All speakers present in the dataset (alphabetical order matches repo).
inline constexpr std::array<std::string_view, 6> kSpeakers = {
    "george", "jackson", "lucas", "nicolas", "theo", "yweweler"
};

// Digit index → English word label.
inline constexpr std::array<std::string_view, 10> kDigitNames = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
};

// WAV filename pattern: {digit}_{speaker}_{trial}.wav
// e.g. "0_jackson_0.wav", "9_yweweler_49.wav"
inline constexpr std::string_view kFilenamePattern = "{digit}_{speaker}_{trial}.wav";

} // namespace nn::dataLoaders::fsdd
