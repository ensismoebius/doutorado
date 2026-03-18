#ifndef EXEC_LOADINGDATA_DEMOPROBEMODEL_HPP
#define EXEC_LOADINGDATA_DEMOPROBEMODEL_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/tensor/Tensor.hpp"

using namespace nn::dataLoaders;

class DemoProbeModel
{
   public:
    [[nodiscard]] auto forward(const nn::Tensor& batch_inputs) const -> nn::Tensor
    {
        constexpr std::size_t eeg_features = ImaginedSpeechSchema_10_1117.eegSignalColumns();
        constexpr std::size_t audio_features = ImaginedSpeechSchema_10_1117.audioSamples();
        constexpr std::size_t stacked_rows = ImaginedSpeechSchema_10_1117.eeg_channels + 1U;
        constexpr std::size_t stacked_concat_features = stacked_rows * audio_features;

        const bool has_stacked_single_sample =
            batch_inputs.rows() == static_cast<int>(stacked_rows) &&
            batch_inputs.cols() == static_cast<int>(audio_features);
        const bool has_concatenated_modalities =
            batch_inputs.cols() == static_cast<int>(stacked_concat_features);
        const bool has_only_eeg = batch_inputs.cols() == eeg_features;
        if (!has_stacked_single_sample && !has_concatenated_modalities && !has_only_eeg)
        {
            throw std::runtime_error(
                "DemoProbeModel expects either EEG-only or EEG+audio input columns.");
        }

        if (has_stacked_single_sample)
        {
            nn::Tensor features(1, 2);

            float audio_abs_sum = 0.0f;
            for (std::size_t c = 0; c < audio_features; ++c)
            {
                audio_abs_sum += std::abs(batch_inputs.at(0, c));
            }

            float eeg_abs_sum = 0.0f;
            for (std::size_t eeg_row = 1; eeg_row < stacked_rows; ++eeg_row)
            {
                for (std::size_t c = 0; c < audio_features; ++c)
                {
                    eeg_abs_sum += std::abs(batch_inputs.at(eeg_row, c));
                }
            }

            features.at(0, 0) = eeg_abs_sum / static_cast<float>(eeg_features);
            features.at(0, 1) = audio_abs_sum / static_cast<float>(audio_features);
            return features;
        }

        nn::Tensor features(batch_inputs.rows(), 2);

        for (std::size_t r = 0; r < batch_inputs.rows(); ++r)
        {
            float eeg_abs_sum = 0.0f;
            float audio_abs_sum = 0.0f;

            if (has_concatenated_modalities)
            {
                for (std::size_t c = 0; c < audio_features; ++c)
                {
                    audio_abs_sum += std::abs(batch_inputs.at(r, c));
                }

                for (std::size_t c = 0; c < eeg_features; ++c)
                {
                    eeg_abs_sum += std::abs(batch_inputs.at(r, audio_features + c));
                }

                features.at(r, 0) = eeg_abs_sum / static_cast<float>(eeg_features);
                features.at(r, 1) = audio_abs_sum / static_cast<float>(audio_features);
            }
            else
            {
                for (std::size_t c = 0; c < eeg_features; ++c)
                {
                    eeg_abs_sum += std::abs(batch_inputs.at(r, c));
                }

                // EEG-only mode keeps audio contribution at zero so output shape
                // remains identical across both input formats.
                features.at(r, 0) = eeg_abs_sum / static_cast<float>(eeg_features);
                features.at(r, 1) = 0.0f;
            }
        }

        return features;
    }
};

#endif // EXEC_LOADINGDATA_DEMOPROBEMODEL_HPP
