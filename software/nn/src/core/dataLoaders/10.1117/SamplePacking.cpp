#include "nn/dataLoaders/10.1117/SamplePacking.hpp"

#include <cstddef>
#include <stdexcept>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/SchemaIndexing.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::schema101117::multimodalInputFeatureColumns;

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

    nn::Tensor input(1, multimodalInputFeatureColumns());

    std::size_t col = 0;
    for (std::size_t ch = 0; ch < ImaginedSpeechSchema_10_1117.eeg_channels; ++ch)
    {
        for (std::size_t s = 0; s < ImaginedSpeechSchema_10_1117.eegSamplesPerChannel(); ++s)
        {
            input.at(0, col++) = eeg.at(ch, s);
        }
    }

    for (std::size_t i = 0; i < ImaginedSpeechSchema_10_1117.audioSamples(); ++i)
    {
        input.at(0, col++) = audio.at(i, 0);
    }

    return input;
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