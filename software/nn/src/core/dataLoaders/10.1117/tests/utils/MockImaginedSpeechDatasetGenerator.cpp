#include "MockImaginedSpeechDatasetGenerator.hpp"

#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/NAMES.hpp"

namespace nn::dataLoaders::test
{
namespace
{
constexpr unsigned int kSeed = 42U;
constexpr double kNoiseSigma = 0.01;

using std::size_t;

auto effective_eeg_columns(
    const nn::dataLoaders::DatasetSchema& schema, EEGCorruptionOptions options) -> size_t
{
    size_t cols = schema.eegTotalColumns();
    if (options.wrong_column_count)
    {
        cols -= 1;
    }
    if (options.missing_labels)
    {
        cols = schema.eegSignalColumns();
    }
    return cols;
}

auto effective_audio_columns(
    const nn::dataLoaders::DatasetSchema& schema, AudioCorruptionOptions options) -> size_t
{
    size_t cols = schema.audioTotalColumns();
    if (options.wrong_column_count)
    {
        cols -= 1;
    }
    if (options.missing_labels)
    {
        cols = schema.audioSamples();
    }
    return cols;
}

void ensure_parent_dir_exists(const std::filesystem::path& path)
{
    const auto parent = path.parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent);
    }
}
} // namespace

void MockImaginedSpeechDatasetGenerator::generateEEGMatFile(
    const std::filesystem::path& path, std::size_t trials)
{
    generateCorruptedEEGMatFile(path, trials, EEGCorruptionOptions{});
}

void MockImaginedSpeechDatasetGenerator::generateAudioMatFile(
    const std::filesystem::path& path, std::size_t trials)
{
    generateCorruptedAudioMatFile(path, trials, AudioCorruptionOptions{});
}

void MockImaginedSpeechDatasetGenerator::generateDataset(
    const std::filesystem::path& directory, std::size_t trials)
{
    std::filesystem::create_directories(directory);
    generateEEGMatFile(directory / "eeg.mat", trials);
    generateAudioMatFile(directory / "audio.mat", trials);
}

void MockImaginedSpeechDatasetGenerator::generateCorruptedEEGMatFile(
    const std::filesystem::path& path, std::size_t trials, EEGCorruptionOptions options)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const size_t rows = trials;
    const size_t cols = effective_eeg_columns(schema, options);

    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("EEG mock generator requires non-zero dimensions.");
    }

    ensure_parent_dir_exists(path);

    std::vector<double> data(rows * cols, 0.0);

    std::mt19937 rng(kSeed);
    std::normal_distribution<double> noise(0.0, kNoiseSigma);

    const size_t channels =
        options.wrong_channel_count ? (schema.eeg_channels - 1) : schema.eeg_channels;
    const size_t samples_per_channel = schema.eegSamplesPerChannel();
    const double fs = static_cast<double>(schema.eeg_sampling_rate);
    const double two_pi = 2.0 * std::acos(-1.0);

    for (size_t trial = 0; trial < rows; ++trial)
    {
        for (size_t ch = 0; ch < channels; ++ch)
        {
            const double freq = 8.0 + static_cast<double>(ch);
            const double phase = static_cast<double>(trial) * 0.1;

            for (size_t s = 0; s < samples_per_channel; ++s)
            {
                const size_t signal_col = (ch * samples_per_channel) + s;
                if (signal_col >= cols)
                {
                    continue;
                }

                const double t = static_cast<double>(s) / fs;
                const double value = std::sin((two_pi * freq * t) + phase) + noise(rng);
                data[(signal_col * rows) + trial] = value;
            }
        }

        if (!options.missing_labels && cols > schema.eegBlinkColumn())
        {
            data[(schema.eegModeColumn() * rows) + trial] = static_cast<double>((trial % 5U) + 1U);
            data[(schema.eegStimulusColumn() * rows) + trial] =
                static_cast<double>((trial % 5U) + 1U);
            data[(schema.eegBlinkColumn() * rows) + trial] =
                static_cast<double>((trial % 10U) + 1U);
        }
    }

    auto file = matioCpp::File::Create(path.string());
    matioCpp::MultiDimensionalArray<double> eeg_matrix(
        nn::dataLoaders::EEG_MAT_VARIABLE_NAME, {rows, cols}, data.data());
    file.write(eeg_matrix);
    file.close();
}

void MockImaginedSpeechDatasetGenerator::generateCorruptedAudioMatFile(
    const std::filesystem::path& path, std::size_t trials, AudioCorruptionOptions options)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const size_t rows = trials;
    const size_t cols = effective_audio_columns(schema, options);

    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Audio mock generator requires non-zero dimensions.");
    }

    ensure_parent_dir_exists(path);

    std::vector<double> data(rows * cols, 0.0);

    std::mt19937 rng(kSeed);
    std::normal_distribution<double> noise(0.0, kNoiseSigma);

    const size_t audio_samples =
        options.wrong_signal_length ? (schema.audioSamples() - 100U) : schema.audioSamples();
    const double fs = static_cast<double>(schema.audio_sampling_rate);
    const double two_pi = 2.0 * std::acos(-1.0);

    for (size_t trial = 0; trial < rows; ++trial)
    {
        for (size_t s = 0; s < audio_samples; ++s)
        {
            if (s >= cols)
            {
                continue;
            }

            const double t = static_cast<double>(s) / fs;
            const double voice_like =
                std::sin(two_pi * 200.0 * t) + 0.5 * std::sin(two_pi * 400.0 * t);
            data[(s * rows) + trial] = voice_like + noise(rng);
        }

        if (!options.missing_labels && cols > schema.audioEEGIndexColumn())
        {
            data[(schema.audioStimulusColumn() * rows) + trial] =
                static_cast<double>((trial % 5U) + 1U);
            data[(schema.audioEEGIndexColumn() * rows) + trial] = static_cast<double>(trial);
        }
    }

    auto file = matioCpp::File::Create(path.string());
    matioCpp::MultiDimensionalArray<double> audio_matrix(
        nn::dataLoaders::AUDIO_MAT_VARIABLE_NAME, {rows, cols}, data.data());
    file.write(audio_matrix);
    file.close();
}

} // namespace nn::dataLoaders::test
