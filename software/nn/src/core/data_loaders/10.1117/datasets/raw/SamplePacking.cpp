/**
 * @file src/core/data_loaders/10.1117/datasets/raw/SamplePacking.cpp
 * @brief Implementation for Samplepacking.
 *

 */

#include "data_loaders/10.1117/datasets/raw/SamplePacking.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "data_loaders/10.1117/schema/Metadata.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using std::size_t;
using std::vector;

namespace
{

template <typename ValueAt>
auto linearResampleWithAccessor(std::size_t source_size, std::size_t target_size, ValueAt value_at)
    -> std::vector<float>
{
    if (target_size == 0)
    {
        return {};
    }

    if (source_size == 0)
    {
        return std::vector<float>(target_size, 0.0f);
    }

    if (source_size == target_size)
    {
        std::vector<float> direct(target_size, 0.0f);
        for (std::size_t i = 0; i < target_size; ++i)
        {
            direct[i] = value_at(i);
        }
        return direct;
    }

    if (source_size == 1)
    {
        return std::vector<float>(target_size, value_at(0));
    }

    if (target_size == 1)
    {
        return std::vector<float>{value_at(0)};
    }

    std::vector<float> result(target_size, 0.0f);
    const double scale =
        static_cast<double>(source_size - 1U) / static_cast<double>(target_size - 1U);

    for (std::size_t i = 0; i < target_size; ++i)
    {
        const double source_pos = static_cast<double>(i) * scale;
        const std::size_t left = static_cast<std::size_t>(std::floor(source_pos));
        const std::size_t right = std::min(left + 1U, source_size - 1U);
        const double alpha = source_pos - static_cast<double>(left);
        result[i] = static_cast<float>((1.0 - alpha) * static_cast<double>(value_at(left)) +
                                       alpha * static_cast<double>(value_at(right)));
    }

    return result;
}

} // namespace

/**
 * Build a single per-sample input tensor that preserves EEG channel
 * separation and places audio as the first row. The final tensor has
 * (1 + eeg_channels) rows and `audio_width` columns.
 * Output row order is fixed as:
 *   [audio; eeg_ch1; eeg_ch2; eeg_ch3; eeg_ch4; eeg_ch5; eeg_ch6]
 */
auto mergeAudioAndEEGSignals(const nn::Tensor& eeg_matrix, const nn::Tensor& audio_vector)
    -> nn::Tensor
{
    // Validate input shapes early to provide clear error messages and
    // to avoid surprising behavior later in the pipeline.
    const size_t eeg_channels = ImaginedSpeechSchema_10_1117.eeg_channels;
    const size_t eeg_channel_width = ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const size_t audio_width = ImaginedSpeechSchema_10_1117.audioSamples();

    if (eeg_matrix.rows() != static_cast<int>(eeg_channels) ||
        eeg_matrix.cols() != static_cast<int>(eeg_channel_width))
    {
        throw std::runtime_error("Unexpected EEG shape. Expected [6x4096].");
    }

    if (audio_vector.rows() != static_cast<int>(audio_width) || audio_vector.cols() != 1)
    {
        throw std::runtime_error("Unexpected Audio shape. Expected [176400x1].");
    }

    // Allocate the output tensor: first row is audio, remaining rows
    // correspond to EEG channels in channel order (ch1..chN).
    // The "+1U" accounts for the audio row at the top of the stack.
    nn::Tensor audio_eeg_merged(eeg_channels + 1U, audio_width);

    const vector<float> audio_resampled = linearResampleWithAccessor( //
        audio_width,                                                  //
        audio_width,                                                  //
        [&](size_t idx) { return audio_vector.at(idx, 0); }           //
    );

    // Copy resampled audio into the first row of the output tensor.
    for (size_t i = 0; i < audio_width; ++i)
    {
        audio_eeg_merged.at(0, i) = audio_resampled[i];
    }

    // For each EEG channel, resample (linearly) from the EEG timescale to
    // the audio timescale and place the result in the corresponding row.
    // Row index `ch + 1` reserves row 0 for audio.
    for (size_t ch = 0; ch < eeg_channels; ++ch)
    {
        const vector<float> eeg_channel_resampled = linearResampleWithAccessor( //
            eeg_channel_width,                                                  //
            audio_width,                                                        //
            [&](size_t idx) { return eeg_matrix.at(ch, idx); }                  //
        );

        for (size_t i = 0; i < audio_width; ++i)
        {
            audio_eeg_merged.at(ch + 1U, i) = eeg_channel_resampled[i];
        }
    }

    return audio_eeg_merged;
}

// extractEegFromAssembledRows and extractAudioFromAssembledRows removed;
// dataset now performs these slices inline in `Dataset101117::collate`.

auto buildInputTensor(const nn::Tensor& eeg, const nn::Tensor& audio) -> nn::Tensor
{
    if (eeg.rows() != ImaginedSpeechSchema_10_1117.eeg_channels ||
        eeg.cols() != ImaginedSpeechSchema_10_1117.eegSamplesPerChannel())
    {
        throw std::runtime_error("Unexpected EEG shape. Expected [6x4096].");
    }

    if (audio.rows() != ImaginedSpeechSchema_10_1117.audioSamples() || audio.cols() != 1)
    {
        throw std::runtime_error("Unexpected Audio shape. Expected [176400x1].");
    }

    return mergeAudioAndEEGSignals(eeg, audio);
}

auto buildTargetTensor(int subject_id, const std::array<int, 3>& eeg_labels, int eeg_index_label)
    -> nn::Tensor
{
    nn::Tensor target(1, 5);
    target.at(0, 0) = static_cast<float>(subject_id);
    target.at(0, 1) = static_cast<float>(eeg_labels[0]);
    target.at(0, 2) = static_cast<float>(eeg_labels[1]);
    target.at(0, 3) = static_cast<float>(eeg_labels[2]);
    target.at(0, 4) = static_cast<float>(eeg_index_label);
    return target;
}