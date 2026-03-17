#include "nn/dataLoaders/10.1117/SamplePacking.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;

namespace
{

auto linearResample(const std::vector<float>& source, std::size_t target_size) -> std::vector<float>
{
    if (target_size == 0)
    {
        return {};
    }

    if (source.empty())
    {
        return std::vector<float>(target_size, 0.0f);
    }

    if (source.size() == target_size)
    {
        return source;
    }

    if (source.size() == 1)
    {
        return std::vector<float>(target_size, source.front());
    }

    if (target_size == 1)
    {
        return std::vector<float>{source.front()};
    }

    std::vector<float> result(target_size, 0.0f);
    const double scale =
        static_cast<double>(source.size() - 1U) / static_cast<double>(target_size - 1U);

    for (std::size_t i = 0; i < target_size; ++i)
    {
        const double source_pos = static_cast<double>(i) * scale;
        const std::size_t left = static_cast<std::size_t>(std::floor(source_pos));
        const std::size_t right = std::min(left + 1U, source.size() - 1U);
        const double alpha = source_pos - static_cast<double>(left);
        result[i] = static_cast<float>((1.0 - alpha) * static_cast<double>(source[left]) +
                                       alpha * static_cast<double>(source[right]));
    }

    return result;
}

} // namespace

/**
 * Builds a channel-preserving multimodal input tensor from raw EEG/audio tensors.
 * Output row order is fixed as:
 *   [audio; eeg_ch1; eeg_ch2; eeg_ch3; eeg_ch4; eeg_ch5; eeg_ch6]
 */
auto buildStackedInputTensorFromRaw(const nn::Tensor& eeg_matrix, const nn::Tensor& audio_column)
    -> nn::Tensor
{
    const std::size_t eeg_channels = ImaginedSpeechSchema_10_1117.eeg_channels;
    const std::size_t eeg_channel_width = ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const std::size_t audio_width = ImaginedSpeechSchema_10_1117.audioSamples();

    if (eeg_matrix.rows() != static_cast<int>(eeg_channels) ||
        eeg_matrix.cols() != static_cast<int>(eeg_channel_width))
    {
        throw std::runtime_error("Unexpected EEG shape. Expected [6x4096].");
    }

    if (audio_column.rows() != static_cast<int>(audio_width) || audio_column.cols() != 1)
    {
        throw std::runtime_error("Unexpected Audio shape. Expected [176400x1].");
    }

    std::vector<float> audio_source(audio_width, 0.0f);
    for (std::size_t i = 0; i < audio_width; ++i)
    {
        audio_source[i] = audio_column.at(i, 0);
    }

    nn::Tensor input(eeg_channels + 1U, audio_width);
    const std::vector<float> audio_resampled = linearResample(audio_source, audio_width);
    for (std::size_t i = 0; i < audio_width; ++i)
    {
        input.at(0, i) = audio_resampled[i];
    }

    for (std::size_t ch = 0; ch < eeg_channels; ++ch)
    {
        std::vector<float> eeg_channel_source(eeg_channel_width, 0.0f);
        for (std::size_t i = 0; i < eeg_channel_width; ++i)
        {
            eeg_channel_source[i] = eeg_matrix.at(ch, i);
        }

        const std::vector<float> eeg_channel_resampled =
            linearResample(eeg_channel_source, audio_width);
        for (std::size_t i = 0; i < audio_width; ++i)
        {
            input.at(ch + 1U, i) = eeg_channel_resampled[i];
        }
    }

    return input;
}

auto extractEegFromAssembledRows(const nn::Tensor& assembled_inputs) -> nn::Tensor
{
    const std::size_t eeg_cols = ImaginedSpeechSchema_10_1117.eegSignalColumns();
    const std::size_t audio_cols = ImaginedSpeechSchema_10_1117.audioSamples();
    if (assembled_inputs.cols() != static_cast<int>(eeg_cols + audio_cols))
    {
        throw std::runtime_error("Unexpected assembled input shape for EEG extraction.");
    }

    nn::Tensor eeg_only(assembled_inputs.rows(), eeg_cols);
    for (std::size_t row = 0; row < static_cast<std::size_t>(assembled_inputs.rows()); ++row)
    {
        for (std::size_t col = 0; col < eeg_cols; ++col)
        {
            eeg_only.at(row, col) = assembled_inputs.at(row, col);
        }
    }

    return eeg_only;
}

auto extractAudioFromAssembledRows(const nn::Tensor& assembled_inputs) -> nn::Tensor
{
    const std::size_t eeg_cols = ImaginedSpeechSchema_10_1117.eegSignalColumns();
    const std::size_t audio_cols = ImaginedSpeechSchema_10_1117.audioSamples();
    if (assembled_inputs.cols() != static_cast<int>(eeg_cols + audio_cols))
    {
        throw std::runtime_error("Unexpected assembled input shape for audio extraction.");
    }

    nn::Tensor audio_only(assembled_inputs.rows(), audio_cols);
    for (std::size_t row = 0; row < static_cast<std::size_t>(assembled_inputs.rows()); ++row)
    {
        for (std::size_t col = 0; col < audio_cols; ++col)
        {
            audio_only.at(row, col) = assembled_inputs.at(row, eeg_cols + col);
        }
    }

    return audio_only;
}

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

    return buildStackedInputTensorFromRaw(eeg, audio);
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