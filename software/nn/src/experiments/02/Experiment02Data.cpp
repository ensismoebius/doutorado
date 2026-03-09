#include "Experiment02Data.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "nn/dataLoaders/mat_file_utils.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/wave/audioFeatureExtraction.h"

auto hanning_window(int length) -> std::vector<double>
{
    return nn::core::wave::hanning_window(length);
}

auto load_eeg_data(const std::string& mat_path) -> std::vector<EEGSample>
{
    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, "Sxx_EEG");
    if (!mat_opt)
    {
        throw std::runtime_error("Failed to load EEG data from " + mat_path);
    }

    nn::Tensor mat = std::move(*mat_opt);
    std::vector<EEGSample> samples;

    for (int i = 0; i < static_cast<int>(mat.rows()); ++i)
    {
        EEGSample sample;
        for (int ch = 0; ch < 6; ++ch)
        {
            std::vector<double> channel(4096);
            for (int s = 0; s < 4096; ++s)
            {
                channel[s] = mat.at(i, ch * 4096 + s);
            }
            sample.channels.push_back(channel);
        }
        sample.modality = static_cast<int>(mat.at(i, 24576));
        sample.stimulus = static_cast<int>(mat.at(i, 24577));
        sample.artifacts = static_cast<int>(mat.at(i, 24578));

        samples.push_back(sample);
    }

    return samples;
}

auto load_audio_data(const std::string& mat_path) -> std::vector<AudioSample>
{
    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, "Sxx_Audio");
    if (!mat_opt)
    {
        throw std::runtime_error("Failed to load Audio data from " + mat_path);
    }

    nn::Tensor mat = std::move(*mat_opt);
    std::vector<AudioSample> samples;

    for (int i = 0; i < static_cast<int>(mat.rows()); ++i)
    {
        AudioSample sample;
        sample.signal.resize(176400);
        for (int s = 0; s < 176400; ++s)
        {
            sample.signal[s] = mat.at(i, s);
        }
        sample.stimulus = static_cast<int>(mat.at(i, 176400));
        sample.eeg_index = static_cast<int>(mat.at(i, 176401));

        samples.push_back(sample);
    }

    return samples;
}

auto extract_windows(const std::vector<EEGSample>& eeg_samples,
                     const std::vector<AudioSample>& audio_samples, double window_duration_sec,
                     double overlap_sec, int eeg_rate, int audio_rate)
    -> std::vector<WindowedSample>
{
    std::vector<WindowedSample> windows;

    for (std::size_t eeg_idx = 0; eeg_idx < eeg_samples.size(); ++eeg_idx)
    {
        const auto& eeg = eeg_samples[eeg_idx];

        const AudioSample* audio = nullptr;
        for (const auto& a : audio_samples)
        {
            if (a.eeg_index == static_cast<int>(eeg_idx))
            {
                audio = &a;
                break;
            }
        }
        if (!audio || eeg.artifacts != 1)
        {
            continue;
        }

        int eeg_window_samples = static_cast<int>(window_duration_sec * eeg_rate);
        int audio_window_samples = static_cast<int>(window_duration_sec * audio_rate);
        int eeg_step = static_cast<int>((window_duration_sec - overlap_sec) * eeg_rate);

        for (int start_eeg = 0; start_eeg + eeg_window_samples <= 4096; start_eeg += eeg_step)
        {
            int start_audio = start_eeg * audio_rate / eeg_rate;
            if (start_audio + audio_window_samples > static_cast<int>(audio->signal.size()))
            {
                break;
            }

            WindowedSample window;
            window.label = eeg.stimulus;

            for (int ch = 0; ch < 6; ++ch)
            {
                for (int s = start_eeg; s < start_eeg + eeg_window_samples; ++s)
                {
                    window.eeg_window.push_back(eeg.channels[ch][s]);
                }
            }

            for (int s = start_audio; s < start_audio + audio_window_samples; ++s)
            {
                window.audio_window.push_back(audio->signal[s]);
            }

            auto hanning_eeg = hanning_window(eeg_window_samples);
            for (std::size_t i = 0; i < window.eeg_window.size(); ++i)
            {
                window.eeg_window[i] *= hanning_eeg[i % eeg_window_samples];
            }

            auto hanning_audio = hanning_window(audio_window_samples);
            for (std::size_t i = 0; i < window.audio_window.size(); ++i)
            {
                window.audio_window[i] *= hanning_audio[i];
            }

            windows.push_back(window);
        }
    }

    return windows;
}
