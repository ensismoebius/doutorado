/**
 * @file src/experiments/waveletAE/WaveletAEData.cpp
 * @brief Implementation for WaveletAEdata.
 *

 */

#include "WaveletAEData.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_loaders/mat_io/mat_file_utils.hpp"
#include "tensor/Tensor.hpp"
#include "wave/audioFeatureExtraction.hpp"

namespace
{
constexpr int kEegChannelCount = 6;
constexpr int kEegSamplesPerChannel = 4096;
constexpr int kAudioSamplesPerRow = 176400;
constexpr int kEegModalityColumn = 24576;
constexpr int kEegStimulusColumn = 24577;
constexpr int kEegArtifactsColumn = 24578;
constexpr int kAudioStimulusColumn = 176400;
constexpr int kAudioEegIndexColumn = 176401;
constexpr int kNoArtifactsFlag = 1;
} // namespace

auto load_eeg_data(const std::string& mat_path) -> std::vector<EEGSample>
{
    auto matrix_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, "Sxx_EEG");
    if (!matrix_opt)
    {
        throw std::runtime_error("Failed to load EEG data from " + mat_path);
    }

    nn::Tensor matrix = std::move(*matrix_opt);
    std::vector<EEGSample> samples;

    for (int row = 0; row < static_cast<int>(matrix.rows()); ++row)
    {
        EEGSample sample;
        for (int channel = 0; channel < kEegChannelCount; ++channel)
        {
            std::vector<double> channel_values(kEegSamplesPerChannel);
            for (int sample_index = 0; sample_index < kEegSamplesPerChannel; ++sample_index)
            {
                channel_values[sample_index] =
                    matrix.at(row, channel * kEegSamplesPerChannel + sample_index);
            }
            sample.channels.push_back(channel_values);
        }
        sample.modality = static_cast<int>(matrix.at(row, kEegModalityColumn));
        sample.stimulus = static_cast<int>(matrix.at(row, kEegStimulusColumn));
        sample.artifacts = static_cast<int>(matrix.at(row, kEegArtifactsColumn));

        samples.push_back(sample);
    }

    return samples;
}

auto load_audio_data(const std::string& mat_path) -> std::vector<AudioSample>
{
    auto matrix_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, "Sxx_Audio");
    if (!matrix_opt)
    {
        throw std::runtime_error("Failed to load Audio data from " + mat_path);
    }

    nn::Tensor matrix = std::move(*matrix_opt);
    std::vector<AudioSample> samples;

    for (int row = 0; row < static_cast<int>(matrix.rows()); ++row)
    {
        AudioSample sample;
        sample.signal.resize(kAudioSamplesPerRow);
        for (int sample_index = 0; sample_index < kAudioSamplesPerRow; ++sample_index)
        {
            sample.signal[sample_index] = matrix.at(row, sample_index);
        }
        sample.stimulus = static_cast<int>(matrix.at(row, kAudioStimulusColumn));
        sample.eeg_index = static_cast<int>(matrix.at(row, kAudioEegIndexColumn));

        samples.push_back(sample);
    }

    return samples;
}

auto extract_windows(const std::vector<EEGSample>& eeg_samples,
    const std::vector<AudioSample>& audio_samples,
    double window_duration_sec,
    double overlap_sec,
    int eeg_rate,
    int audio_rate) -> std::vector<WindowedSample>
{
    if (window_duration_sec <= 0.0)
    {
        throw std::invalid_argument("window_duration_sec must be > 0");
    }
    if (overlap_sec < 0.0 || overlap_sec >= window_duration_sec)
    {
        throw std::invalid_argument(
            "overlap_sec must be >= 0 and strictly less than window_duration_sec");
    }
    if (eeg_rate <= 0 || audio_rate <= 0)
    {
        throw std::invalid_argument("eeg_rate and audio_rate must be > 0");
    }

    std::vector<WindowedSample> windows;

    const int eeg_window_samples = static_cast<int>(window_duration_sec * eeg_rate);
    const int audio_window_samples = static_cast<int>(window_duration_sec * audio_rate);
    const int eeg_step = static_cast<int>((window_duration_sec - overlap_sec) * eeg_rate);

    if (eeg_window_samples <= 0 || audio_window_samples <= 0 || eeg_step <= 0)
    {
        throw std::invalid_argument("computed window/step sizes must be > 0");
    }

    const auto hanning_eeg = nn::core::wave::hanning_window(eeg_window_samples);
    const auto hanning_audio = nn::core::wave::hanning_window(audio_window_samples);

    for (std::size_t eeg_idx = 0; eeg_idx < eeg_samples.size(); ++eeg_idx)
    {
        const auto& eeg = eeg_samples[eeg_idx];

        const auto audio_it = std::find_if(audio_samples.begin(),
            audio_samples.end(),
            [eeg_idx](const auto& a) { return a.eeg_index == static_cast<int>(eeg_idx); });
        const AudioSample* audio = (audio_it != audio_samples.end()) ? &(*audio_it) : nullptr;
        if (!audio || eeg.artifacts != kNoArtifactsFlag)
        {
            continue;
        }

        for (int start_eeg = 0; start_eeg + eeg_window_samples <= kEegSamplesPerChannel;
            start_eeg += eeg_step)
        {
            int start_audio = start_eeg * audio_rate / eeg_rate;
            if (start_audio + audio_window_samples > static_cast<int>(audio->signal.size()))
            {
                break;
            }

            WindowedSample window;
            window.label = eeg.stimulus;

            for (int ch = 0; ch < kEegChannelCount; ++ch)
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

            std::transform(window.eeg_window.begin(),
                window.eeg_window.end(),
                window.eeg_window.begin(),
                [idx = std::size_t{0}, eeg_window_samples, &hanning_eeg](double value) mutable
                {
                    const double weighted = value * hanning_eeg[idx % eeg_window_samples];
                    ++idx;
                    return weighted;
                });

            std::transform(window.audio_window.begin(),
                window.audio_window.end(),
                hanning_audio.begin(),
                window.audio_window.begin(),
                [](double sample, double hann) { return sample * hann; });

            windows.push_back(window);
        }
    }

    return windows;
}

void generate_synthetic_samples(std::vector<EEGSample>& eeg_samples,
    std::vector<AudioSample>& audio_samples,
    int sample_count,
    int random_seed)
{
    eeg_samples.resize(sample_count);
    audio_samples.resize(sample_count);

    std::mt19937 random_engine(random_seed);
    std::normal_distribution<double> gaussian_dist(0.0, 1.0);

    for (int sample_index = 0; sample_index < sample_count; ++sample_index)
    {
        EEGSample eeg_sample;
        eeg_sample.channels.resize(kEegChannelCount, std::vector<double>(kEegSamplesPerChannel));
        for (auto& channel : eeg_sample.channels)
        {
            std::generate(
                channel.begin(), channel.end(), [&]() { return gaussian_dist(random_engine); });
        }
        eeg_sample.modality = 1;
        eeg_sample.stimulus = (sample_index % 5) + 1;
        eeg_sample.artifacts = kNoArtifactsFlag;
        eeg_samples[sample_index] = eeg_sample;

        AudioSample audio_sample;
        audio_sample.signal.resize(kAudioSamplesPerRow);
        std::generate(audio_sample.signal.begin(),
            audio_sample.signal.end(),
            [&]() { return gaussian_dist(random_engine); });
        audio_sample.stimulus = eeg_sample.stimulus;
        audio_sample.eeg_index = sample_index;
        audio_samples[sample_index] = audio_sample;
    }
}
