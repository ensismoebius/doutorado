#pragma once
// Metadata.hpp — FSDD dataset constants (DOI: 10.5281/zenodo.1342401).
//
// Free Spoken Digit Dataset (FSDD) — Jakobovski et al.
// https://github.com/Jakobovski/free-spoken-digit-dataset
//
// 6 speakers × 10 digits × 50 trials = 3 000 WAV files.
// Files named: {digit}_{speaker}_{trial}.wav  (e.g. "0_jackson_0.wav")

#include <cstddef>

namespace nn::dataLoaders::fsdd
{

constexpr int    kDigitCount              = 10;    // digits 0–9
constexpr int    kSpeakerCount            = 6;
constexpr int    kTrialsPerSpeakerDigit   = 50;
constexpr int    kTotalFiles              = kDigitCount * kSpeakerCount * kTrialsPerSpeakerDigit; // 3000
constexpr int    kSampleRate              = 8000;  // Hz
constexpr int    kChannels                = 1;     // mono

} // namespace nn::dataLoaders::fsdd
