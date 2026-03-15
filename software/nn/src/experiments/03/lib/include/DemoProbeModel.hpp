#ifndef EXEC_LOADINGDATA_DEMOPROBEMODEL_HPP
#define EXEC_LOADINGDATA_DEMOPROBEMODEL_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/tensor/Tensor.hpp"

class DemoProbeModel
{
   public:
    [[nodiscard]] auto forward(const nn::Tensor& batch_inputs) const -> nn::Tensor
    {
        constexpr std::size_t eeg_features =
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns();
        constexpr std::size_t input_features =
            eeg_features + nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();

        const bool has_concatenated_modalities = batch_inputs.cols() == input_features;
        const bool has_only_eeg = batch_inputs.cols() == eeg_features;
        if (!has_concatenated_modalities && !has_only_eeg)
        {
            throw std::runtime_error(
                "DemoProbeModel expects either EEG-only or EEG+audio input columns.");
        }

        nn::Tensor features(batch_inputs.rows(), 2);

        for (std::size_t r = 0; r < batch_inputs.rows(); ++r)
        {
            float eeg_abs_sum = 0.0f;
            for (std::size_t c = 0; c < eeg_features; ++c)
            {
                eeg_abs_sum += std::abs(batch_inputs.at(r, c));
            }

            float audio_abs_sum = 0.0f;
            if (has_concatenated_modalities)
            {
                for (std::size_t c = eeg_features; c < input_features; ++c)
                {
                    audio_abs_sum += std::abs(batch_inputs.at(r, c));
                }
            }
            // EEG-only mode keeps audio contribution at zero so output shape
            // remains identical across both input formats.

            features.at(r, 0) = eeg_abs_sum / static_cast<float>(eeg_features);
            features.at(r, 1) =
                audio_abs_sum /
                static_cast<float>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples());
        }

        return features;
    }
};

#endif // EXEC_LOADINGDATA_DEMOPROBEMODEL_HPP
