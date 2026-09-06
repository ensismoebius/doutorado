/**
 * @file src/experiments/thesis/lib/include/ThesisSample.hpp
 * @brief ThesisSample struct (extracted from ThesisDataset.hpp).
 */

#pragma once

#include <string>

#include "tensor/Tensor.hpp"

namespace thesis
{

// One raw sample from a single subject: audio + EEG tensors plus stimulus label.
struct ThesisSample
{
    nn::Tensor audio; // shape (N_audio_samples, 1)
    nn::Tensor eeg;   // shape (N_eeg_channels, N_eeg_samples)
    int stimulus = 0; // word/vowel index
    int subject_id = 0;
    std::string text_phrase; // e.g. "arriba", "a", etc.
};

} // namespace thesis
