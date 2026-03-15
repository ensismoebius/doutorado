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

auto buildInputTensorFromFlattenedRows(const nn::Tensor& eeg_row, const nn::Tensor& audio_row)
    -> nn::Tensor
{
    if (eeg_row.rows() != 1 ||
        eeg_row.cols() != static_cast<int>(ImaginedSpeechSchema_10_1117.eegSignalColumns()))
    {
        throw std::runtime_error("Unexpected flattened EEG shape. Expected [1x24576].");
    }

    if (audio_row.rows() != 1 ||
        audio_row.cols() != static_cast<int>(ImaginedSpeechSchema_10_1117.audioSamples()))
    {
        throw std::runtime_error("Unexpected flattened Audio shape. Expected [1x176400].");
    }

    const std::size_t common_width = ImaginedSpeechSchema_10_1117.eegSignalColumns();

    std::vector<float> eeg_source(common_width, 0.0f);
    for (std::size_t i = 0; i < common_width; ++i)
    {
        eeg_source[i] = eeg_row.at(0, i);
    }

    const std::size_t audio_width = ImaginedSpeechSchema_10_1117.audioSamples();
    std::vector<float> audio_source(audio_width, 0.0f);
    for (std::size_t i = 0; i < audio_width; ++i)
    {
        audio_source[i] = audio_row.at(0, i);
    }

    // Both modalities are projected to a common temporal grid (linear interpolation)
    // and then stacked vertically as [audio; eeg] before flattening to row-major.
    const std::vector<float> audio_resampled = linearResample(audio_source, common_width);
    const std::vector<float> eeg_resampled = linearResample(eeg_source, common_width);

    nn::Tensor input(1, common_width * 2U);
    for (std::size_t i = 0; i < common_width; ++i)
    {
        input.at(0, i) = audio_resampled[i];
        input.at(0, common_width + i) = eeg_resampled[i];
    }

    return input;
}

auto flattenAudioColumnToRow(const nn::Tensor& audio_column) -> nn::Tensor
{
    if (audio_column.cols() != 1)
    {
        throw std::runtime_error("Unexpected audio shape for flatten. Expected [Nx1].");
    }

    nn::Tensor audio_row(1, audio_column.rows());
    for (std::size_t i = 0; i < static_cast<std::size_t>(audio_column.rows()); ++i)
    {
        audio_row.at(0, i) = audio_column.at(i, 0);
    }
    return audio_row;
}

auto flattenEegMatrixToRow(const nn::Tensor& eeg_matrix) -> nn::Tensor
{
    nn::Tensor eeg_row(1, eeg_matrix.rows() * eeg_matrix.cols());
    std::size_t eeg_col = 0;
    for (std::size_t row = 0; row < static_cast<std::size_t>(eeg_matrix.rows()); ++row)
    {
        for (std::size_t col = 0; col < static_cast<std::size_t>(eeg_matrix.cols()); ++col)
        {
            eeg_row.at(0, eeg_col++) = eeg_matrix.at(row, col);
        }
    }
    return eeg_row;
}

auto extractEegFromConcatenatedRows(const nn::Tensor& concatenated_inputs) -> nn::Tensor
{
    const std::size_t eeg_cols = ImaginedSpeechSchema_10_1117.eegSignalColumns();
    const std::size_t audio_cols = ImaginedSpeechSchema_10_1117.audioSamples();
    if (concatenated_inputs.cols() != static_cast<int>(eeg_cols + audio_cols))
    {
        throw std::runtime_error("Unexpected concatenated input shape for EEG extraction.");
    }

    nn::Tensor eeg_only(concatenated_inputs.rows(), eeg_cols);
    for (std::size_t row = 0; row < static_cast<std::size_t>(concatenated_inputs.rows()); ++row)
    {
        for (std::size_t col = 0; col < eeg_cols; ++col)
        {
            eeg_only.at(row, col) = concatenated_inputs.at(row, col);
        }
    }

    return eeg_only;
}

auto extractAudioFromConcatenatedRows(const nn::Tensor& concatenated_inputs) -> nn::Tensor
{
    const std::size_t eeg_cols = ImaginedSpeechSchema_10_1117.eegSignalColumns();
    const std::size_t audio_cols = ImaginedSpeechSchema_10_1117.audioSamples();
    if (concatenated_inputs.cols() != static_cast<int>(eeg_cols + audio_cols))
    {
        throw std::runtime_error("Unexpected concatenated input shape for audio extraction.");
    }

    nn::Tensor audio_only(concatenated_inputs.rows(), audio_cols);
    for (std::size_t row = 0; row < static_cast<std::size_t>(concatenated_inputs.rows()); ++row)
    {
        for (std::size_t col = 0; col < audio_cols; ++col)
        {
            audio_only.at(row, col) = concatenated_inputs.at(row, eeg_cols + col);
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

    const nn::Tensor eeg_row = flattenEegMatrixToRow(eeg);
    const nn::Tensor audio_row = flattenAudioColumnToRow(audio);

    return buildInputTensorFromFlattenedRows(eeg_row, audio_row);
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